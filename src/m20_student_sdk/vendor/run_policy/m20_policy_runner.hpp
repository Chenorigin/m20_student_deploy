/**
 * @file m20_policy_runner.hpp
 * @brief m20_policy_runner
 * @author Bo (Percy) Peng
 * @version 1.0
 * @date 2025-11-07
 * 
 * @copyright Copyright (c) 2025  DeepRobotics
 * 
 */

#pragma once
#define PI 3.14159265358979323846

#include "policy_runner_base.hpp"
#include "v11_riser_extractor.hpp"
#include <ctime>
#include <array>
#include <cmath>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <utility>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#ifdef M20_HAS_REALSENSE2
#include "realsense_depth_source.hpp"
#endif

class M20PolicyRunner : public PolicyRunnerBase {
private:
    VecXf kp_, kd_;
    VecXf dof_default_eigen_policy, dof_default_eigen_robot;
    Vec3f max_cmd_vel_, gravity_direction = Vec3f(0., 0., -1.);
    VecXf dof_pos_default_;
    timespec system_time;

    const int motor_num = 16;
    static constexpr int proprioception_dim = 57;
    static constexpr int lidar_v7_observation_dim = 2010;
    static constexpr int lidar_v7_frame_dim = 402;
    static constexpr int depth_history_length = 3;
    static constexpr int depth_height = 48;
    static constexpr int depth_width = 64;
    static constexpr int depth_tensor_dim = depth_history_length * depth_height * depth_width;
    static constexpr int depth_status_dim = 2;
    static constexpr int depth_message_dim = depth_tensor_dim + depth_status_dim;
    static constexpr int depth_actor_observation_dim = proprioception_dim + depth_status_dim;
    static constexpr int v11_feature_dim = 6;
    static constexpr int v11_actor_observation_dim = proprioception_dim + v11_feature_dim;
    static constexpr float perceptive_stop_age = 0.5f;  // 0.25 s when status_max_age is 0.5 s
    static constexpr float zero_command_wheel_stop_threshold = 0.08f;

    enum class ObservationMode {
        kFlat,
        kDepthDualInput,
        kV11TwoRiser,
    };

    ObservationMode observation_mode_ = ObservationMode::kFlat;
    m20::v11::FeatureMode v11_feature_mode_ = m20::v11::FeatureMode::PositiveRisers;
    int observation_dim_ = proprioception_dim;
    int lidar_observation_dim_ = 0;
    const int action_dim = 16;
    float agent_timestep = 0.02;
    float current_time;
    bool is_fallen = true;

    VecXf joint_pos_rl = VecXf(action_dim);// in rl squenece
    VecXf joint_vel_rl = VecXf(action_dim);
    
    const std::string policy_path_;

    float omega_scale_ = 0.25;
    float dof_vel_scale_ = 0.05;
    VecXf imu_w_eigen, base_acc_eigen, motor_p_eigen, motor_v_eigen,
          current_action_eigen, last_action_eigen, current_observation_, projected_gravity,
          tmp_action_eigen, depth_input_;

    RobotAction robot_action;
    std::vector<std::string> robot_order = {
        "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint", "fl_wheel_joint",
        "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint", "fr_wheel_joint",
        "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint", "hl_wheel_joint",
        "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint", "hr_wheel_joint"};


    std::vector<std::string> policy_order = {
        "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
        "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
        "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
        "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
        "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
    };


    std::vector<float> action_scale_robot = {0.125, 0.25, 0.25, 5,
                                             0.125, 0.25, 0.25, 5,
                                             0.125, 0.25, 0.25, 5,
                                             0.125, 0.25, 0.25, 5};


    Ort::SessionOptions session_options_;
    Ort::Session session_{nullptr};
    
    Ort::Env env_;
    std::vector<int> robot2policy_idx, policy2robot_idx;

    const char* flat_input_names_[1] = {"obs"};
    const char* depth_input_names_[2] = {"obs", "depth"};
    const char* output_names_[1] = {"actions"};
    VecXf command;
    Ort::MemoryInfo memory_info{nullptr};
    std::array<int64_t, 2> input_observation_shape_ = {1, proprioception_dim};
    const std::array<int64_t, 4> input_depth_shape_ = {
        1, depth_history_length, depth_height, depth_width};
    bool warned_missing_perception_ = false;
    bool warned_invalid_observation_ = false;
    bool warned_invalid_action_ = false;
    bool debug_policy_io_ = false;
    bool zero_command_wheel_stop_ = true;
    int debug_policy_io_count_ = 0;
    bool use_direct_realsense_ = false;
#ifdef M20_HAS_REALSENSE2
    std::unique_ptr<RealSenseDepthSource> realsense_depth_source_;
#endif
    
    float time_step = 0.;
    int stop_count = 1000;

    bool IsV11ObservationDeployable(const VecXf& observation) const {
        if (observation.size() != v11_feature_dim) {
            return false;
        }
        m20::v11::Feature feature{};
        std::copy_n(observation.data(), v11_feature_dim, feature.begin());
        return m20::v11::IsValidFeature(feature, v11_feature_mode_);
    }

public:
    M20PolicyRunner(const std::string &policy_name, const std::string &policy_path) :
            PolicyRunnerBase(policy_name), policy_path_(policy_path),env_(ORT_LOGGING_LEVEL_WARNING, "M20PolicyRunner"),
            session_options_{},
            session_{nullptr},
            memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

        dof_default_eigen_policy.setZero(action_dim);
        dof_default_eigen_robot.setZero(action_dim);
        dof_default_eigen_policy << 0.0, -0.6,  1.0,
                                    0.0, -0.6,  1.0,
                                    0.0,  0.6, -1.0,
                                    0.0,  0.6, -1.0,
                                    0.0, 0.0, 0.0, 0.0;
        dof_default_eigen_robot << 0.0, -0.6,  1.0, 0.0,
                                   0.0, -0.6,  1.0, 0.0,
                                   0.0,  0.6, -1.0, 0.0,
                                   0.0,  0.6, -1.0, 0.0;
        SetDecimation(4);
        session_options_.SetIntraOpNumThreads(4);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        if (access(policy_path_.c_str(), F_OK) != 0) {
            std::cerr << "Model file not found: " << policy_path_ << std::endl;
            throw std::runtime_error("Model file missing");
            }

        // 加载模型
        session_ = Ort::Session(env_, policy_path_.c_str(), session_options_);
        Ort::AllocatorWithDefaultOptions allocator;
        const size_t input_count = session_.GetInputCount();
        std::unordered_map<std::string, size_t> input_indices;
        for (size_t index = 0; index < input_count; ++index) {
            auto input_name = session_.GetInputNameAllocated(index, allocator);
            input_indices.emplace(input_name.get(), index);
        }
        if (input_count == 1 && input_indices.count("obs") == 1) {
            const auto input_shape = session_.GetInputTypeInfo(input_indices.at("obs"))
                                         .GetTensorTypeAndShapeInfo().GetShape();
            if (input_shape.size() != 2 || (input_shape[0] != 1 && input_shape[0] != -1) ||
                input_shape[1] < proprioception_dim) {
                throw std::runtime_error(
                    "M20 flat policy input 'obs' must be [1, N] or [dynamic_batch, N] "
                    "with static N >= 57");
            }
            observation_dim_ = static_cast<int>(input_shape[1]);
            if (observation_dim_ == v11_actor_observation_dim) {
                observation_mode_ = ObservationMode::kV11TwoRiser;
                lidar_observation_dim_ = 0;
                std::cout << "Policy input: [1, 63], V11 two-riser features: 6" << std::endl;
            } else {
                lidar_observation_dim_ = observation_dim_ - proprioception_dim;
                std::cout << "Policy input: [1, " << observation_dim_ << "], proprioception: "
                          << proprioception_dim << ", lidar: " << lidar_observation_dim_ << std::endl;
            }
            input_observation_shape_ = {1, observation_dim_};
        } else if (input_count == 2 && input_indices.count("obs") == 1 &&
                   input_indices.count("depth") == 1) {
            const auto obs_shape = session_.GetInputTypeInfo(input_indices.at("obs"))
                                       .GetTensorTypeAndShapeInfo().GetShape();
            const auto depth_shape = session_.GetInputTypeInfo(input_indices.at("depth"))
                                         .GetTensorTypeAndShapeInfo().GetShape();
            if (obs_shape.size() != 2 || (obs_shape[0] != 1 && obs_shape[0] != -1) ||
                obs_shape[1] != depth_actor_observation_dim) {
                throw std::runtime_error("Depth policy input 'obs' must be [batch, 59]");
            }
            if (depth_shape.size() != 4 || (depth_shape[0] != 1 && depth_shape[0] != -1) ||
                depth_shape[1] != depth_history_length || depth_shape[2] != depth_height ||
                depth_shape[3] != depth_width) {
                throw std::runtime_error("Depth V2 policy input 'depth' must be [batch, 3, 48, 64]");
            }
            observation_mode_ = ObservationMode::kDepthDualInput;
            observation_dim_ = depth_actor_observation_dim;
            lidar_observation_dim_ = 0;
            input_observation_shape_ = {1, observation_dim_};
            std::cout << "Policy inputs: obs=[1, 59], depth=[1, 3, 48, 64]; "
                         "depth message: 9218 floats" << std::endl;
        } else {
            throw std::runtime_error(
                "M20 policy must expose either one input named 'obs' or two inputs named "
                "'obs' and 'depth'");
        }
        if (observation_mode_ == ObservationMode::kV11TwoRiser) {
            static constexpr const char* kLegacyContract = "v11_positive_risers_v1";
            static constexpr const char* kSignedContract = "v11_signed_edges_v2";
            static constexpr const char* kSignedRearAxleContract =
                "v11_signed_edges_rear_axle_v3";
            const char* configured_contract = std::getenv("M20_V11_FEATURE_CONTRACT");
            const std::string expected_contract =
                configured_contract == nullptr ? kLegacyContract : configured_contract;
            if (expected_contract != kLegacyContract && expected_contract != kSignedContract &&
                expected_contract != kSignedRearAxleContract) {
                throw std::runtime_error(
                    "M20_V11_FEATURE_CONTRACT must be v11_positive_risers_v1 or "
                    "v11_signed_edges_v2 or v11_signed_edges_rear_axle_v3");
            }
            v11_feature_mode_ = expected_contract == kSignedRearAxleContract
                ? m20::v11::FeatureMode::SignedEdgesRearAxle
                : (expected_contract == kSignedContract
                    ? m20::v11::FeatureMode::SignedEdges
                    : m20::v11::FeatureMode::PositiveRisers);

            auto metadata = session_.GetModelMetadata();
            auto embedded_contract = metadata.LookupCustomMetadataMapAllocated(
                "m20_feature_contract", allocator);
            if (embedded_contract != nullptr) {
                const std::string actual_contract = embedded_contract.get();
                if (actual_contract != expected_contract) {
                    throw std::runtime_error(
                        "V11 ONNX feature contract mismatch: expected " + expected_contract +
                        ", model embeds " + actual_contract);
                }
            } else if (expected_contract != kLegacyContract) {
                throw std::runtime_error(
                    "signed-edge V11 policy is missing required ONNX metadata "
                    "m20_feature_contract");
            }
            std::cout << "V11 feature contract: " << expected_contract
                      << (embedded_contract == nullptr ? " (legacy model without metadata)"
                                                       : " (verified in ONNX metadata)")
                      << std::endl;
        }
        if (session_.GetOutputCount() != 1) {
            throw std::runtime_error("M20 policy must expose exactly one output named 'actions'");
        }
        auto output_name = session_.GetOutputNameAllocated(0, allocator);
        if (std::string(output_name.get()) != "actions") {
            throw std::runtime_error("M20 policy output must be named 'actions'");
        }
        const auto output_shape = session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (output_shape.size() != 2 || (output_shape[0] != 1 && output_shape[0] != -1) ||
            output_shape[1] != action_dim) {
            throw std::runtime_error("M20 policy output must be [1, 16] or [dynamic_batch, 16]");
        }
        if (observation_mode_ == ObservationMode::kDepthDualInput) {
            const char* source = std::getenv("M20_DEPTH_SOURCE");
            const std::string depth_source = source == nullptr ? "ros2" : source;
            if (depth_source == "realsense") {
#ifdef M20_HAS_REALSENSE2
                use_direct_realsense_ = true;
                realsense_depth_source_ = std::make_unique<RealSenseDepthSource>();
                std::cout << "Depth source: in-process librealsense2 (no depth ROS topic)"
                          << std::endl;
#else
                throw std::runtime_error(
                    "M20_DEPTH_SOURCE=realsense requested, but rl_deploy was built without "
                    "BUILD_REALSENSE_DEPTH_SOURCE=ON");
#endif
            } else if (depth_source != "ros2") {
                throw std::runtime_error(
                    "M20_DEPTH_SOURCE must be either 'realsense' or 'ros2'");
            }
        }
        kp_ = Vec4f(80, 80, 80, 0.).replicate(4, 1);
        kd_ = Vec4f(2, 2, 2, 0.6).replicate(4, 1);
        
        robot2policy_idx = generate_permutation(robot_order, policy_order);
        policy2robot_idx = generate_permutation(policy_order, robot_order);
        // for (int i = 0; i < action_dim; ++i){
        //     std::cout << "robot2policy_idx[" << i << "]: " << robot2policy_idx[i] << std::endl;
        //     std::cout << "policy2robot_idx[" << i << "]: " << policy2robot_idx[i] << std::endl;
        // }

        robot_action.kp = kp_;
        robot_action.kd = kd_;
        robot_action.tau_ff = VecXf::Zero(motor_num);
        robot_action.goal_joint_pos = VecXf::Zero(motor_num);
        robot_action.goal_joint_vel = VecXf::Zero(motor_num);


        current_observation_.setZero(observation_dim_);
        depth_input_.setZero(depth_tensor_dim);
        last_action_eigen.setZero(action_dim);
        tmp_action_eigen.setZero(action_dim);
        current_action_eigen.setZero(action_dim);
        warned_missing_perception_ = false;
        warned_invalid_observation_ = false;
        warned_invalid_action_ = false;
        const char* debug_env = std::getenv("M20_POLICY_DEBUG");
        debug_policy_io_ = debug_env != nullptr && debug_env[0] != '\0' && debug_env[0] != '0';
        const char* wheel_stop_env = std::getenv("M20_ZERO_CMD_WHEEL_STOP");
        zero_command_wheel_stop_ = !(wheel_stop_env != nullptr && wheel_stop_env[0] == '0');

        memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    }

    ~M20PolicyRunner() override = default;

    std::vector<int> generate_permutation(
        const std::vector<std::string>& from, 
        const std::vector<std::string>& to, 
        int default_index = 0) 
    {
        std::unordered_map<std::string, int> idx_map;
        for (int i = 0; i < from.size(); ++i) {
            idx_map[from[i]] = i;
        }

        std::vector<int> perm;
        for (const auto& name : to) {
            auto it = idx_map.find(name);
            if (it != idx_map.end()) {
                perm.push_back(it->second);
            } else {
                perm.push_back(default_index);  // 如果找不到，就填默认值
            }
        }

        return perm;
    }

    void DisplayPolicyInfo(){}

    int GetObservationDim() const {
        return observation_dim_;
    }

    int GetLidarObservationDim() const {
        return lidar_observation_dim_;
    }

    int GetDepthObservationDim() const {
        return observation_mode_ == ObservationMode::kDepthDualInput ? depth_message_dim : 0;
    }

    int GetV11ObservationDim() const {
        return observation_mode_ == ObservationMode::kV11TwoRiser ? v11_feature_dim : 0;
    }

    bool IsDepthPolicy() const {
        return observation_mode_ == ObservationMode::kDepthDualInput;
    }

    void OnEnter() {
        run_cnt_ = 0;
        cmd_vel_input_.setZero();
        last_action_eigen.setZero(action_dim);
        tmp_action_eigen.setZero(action_dim);
        motor_p_eigen.setZero(12);
        motor_v_eigen.setZero(motor_num);
        warned_missing_perception_ = false;
        warned_invalid_observation_ = false;
        warned_invalid_action_ = false;
        debug_policy_io_count_ = 0;
    }

    VecXf Onnx_infer(VecXf& current_observation){
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            current_observation.data(),
            current_observation.size(),
            input_observation_shape_.data(),
            input_observation_shape_.size()
        );

        std::vector<Ort::Value> inputs;
        inputs.emplace_back(std::move(input_tensor));

        std::vector<Ort::Value> outputs;
        if (observation_mode_ == ObservationMode::kDepthDualInput) {
            Ort::Value depth_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                depth_input_.data(),
                depth_input_.size(),
                input_depth_shape_.data(),
                input_depth_shape_.size()
            );
            inputs.emplace_back(std::move(depth_tensor));
            outputs = session_.Run(
                Ort::RunOptions{nullptr}, depth_input_names_, inputs.data(), 2,
                output_names_, 1);
        } else {
            outputs = session_.Run(
                Ort::RunOptions{nullptr}, flat_input_names_, inputs.data(), 1,
                output_names_, 1);
        }

        float* action_data = outputs[0].GetTensorMutableData<float>();
        Eigen::Map<Eigen::VectorXf> action_map(action_data, action_dim);
        return VecXf(action_map);  // 返回一个Eigen向量的副本
    }

    RobotAction getRobotAction(const RobotBasicState &ro, const UserCommand &uc) {

        bool perception_ready = true;
        std::string perception_error;
        std::string perception_topic;
        int expected_perception_dim = 0;
        float perception_transport_age = 0.0f;
        VecXf direct_depth_observation;
        const VecXf* selected_depth_observation = &ro.depth_observation;
        bool selected_depth_valid = ro.depth_valid;
#ifdef M20_HAS_REALSENSE2
        if (observation_mode_ == ObservationMode::kDepthDualInput && use_direct_realsense_) {
            const auto snapshot = realsense_depth_source_->GetSnapshot();
            perception_transport_age = snapshot.sample_age_seconds;
            selected_depth_valid = snapshot.ready;
            if (snapshot.sample_count > 0) {
                direct_depth_observation.resize(depth_message_dim);
                std::copy(
                    snapshot.observation.begin(), snapshot.observation.end(),
                    direct_depth_observation.data());
                selected_depth_observation = &direct_depth_observation;
            }
        }
#endif
        if (observation_mode_ == ObservationMode::kV11TwoRiser) {
            perception_topic = "/M20_V11_OBS";
            expected_perception_dim = v11_feature_dim;
            perception_transport_age = ro.v11_age_seconds;
            if (!ro.v11_valid) {
                perception_ready = false;
                perception_error = "message missing, invalid, or transport age exceeded";
            } else if (ro.v11_observation.size() != v11_feature_dim) {
                perception_ready = false;
                perception_error = "feature length mismatch";
            } else if (!ro.v11_observation.allFinite()) {
                perception_ready = false;
                perception_error = "feature contains NaN/Inf";
            } else if (!IsV11ObservationDeployable(ro.v11_observation)) {
                perception_ready = false;
                perception_error = "B/H/D feature is outside its deployable range";
            }
        } else if (observation_mode_ == ObservationMode::kDepthDualInput) {
            perception_topic = use_direct_realsense_ ?
                "in-process RealSense D435i" : "/M20_DEPTH_OBS";
            expected_perception_dim = depth_message_dim;
            if (!use_direct_realsense_) {
                perception_transport_age = ro.depth_age_seconds;
            }
            if (!selected_depth_valid) {
                perception_ready = false;
                perception_error = "frame missing, invalid, or age exceeded";
            } else if (selected_depth_observation->size() != depth_message_dim) {
                perception_ready = false;
                perception_error = "depth length mismatch";
            } else if (!selected_depth_observation->allFinite()) {
                perception_ready = false;
                perception_error = "depth contains NaN/Inf";
            } else if (selected_depth_observation->head(depth_tensor_dim).minCoeff() < -1.0001f ||
                       selected_depth_observation->head(depth_tensor_dim).maxCoeff() > 1.0001f) {
                perception_ready = false;
                perception_error = "normalized depth is outside [-1, 1]";
            } else {
                const float valid_fraction = (*selected_depth_observation)(depth_tensor_dim);
                const float normalized_age = (*selected_depth_observation)(depth_tensor_dim + 1);
                if (valid_fraction < 0.0f || valid_fraction > 1.0f ||
                    normalized_age < 0.0f || normalized_age > 1.0f) {
                    perception_ready = false;
                    perception_error = "depth status is outside [0, 1]";
                } else if (normalized_age >= perceptive_stop_age) {
                    perception_ready = false;
                    perception_error = "newest depth is stale for at least 0.25 s";
                }
            }
        } else if (lidar_observation_dim_ > 0) {
            perception_topic = "/M20_LIDAR_OBS";
            expected_perception_dim = lidar_observation_dim_;
            perception_transport_age = ro.lidar_age_seconds;
            if (!ro.lidar_valid) {
                perception_ready = false;
                perception_error = "message missing, invalid, or transport age exceeded";
            } else if (ro.lidar_observation.size() != lidar_observation_dim_) {
                perception_ready = false;
                perception_error = "message length mismatch";
            } else if (!ro.lidar_observation.allFinite()) {
                perception_ready = false;
                perception_error = "message contains NaN/Inf";
            } else if (lidar_observation_dim_ == lidar_v7_observation_dim) {
                const int newest_status = 4 * lidar_v7_frame_dim + 400;
                const float valid_fraction = ro.lidar_observation(newest_status);
                const float normalized_age = ro.lidar_observation(newest_status + 1);
                if (valid_fraction < 0.0f || valid_fraction > 1.0f ||
                    normalized_age < 0.0f || normalized_age > 1.0f) {
                    perception_ready = false;
                    perception_error = "V7 status is outside [0, 1]";
                } else if (normalized_age >= perceptive_stop_age) {
                    perception_ready = false;
                    perception_error = "V7 newest scan is stale for at least 0.25 s";
                }
            }
        }
        if (!perception_ready) {
            if (!warned_missing_perception_) {
                std::cerr << "Perceptive policy is waiting for " << perception_topic << " with "
                          << expected_perception_dim << " floats (" << perception_error
                          << ", transport age=" << perception_transport_age
                          << " s); holding the default pose and stopping wheels." << std::endl;
                warned_missing_perception_ = true;
            }
            robot_action.goal_joint_pos = dof_default_eigen_robot;
            robot_action.goal_joint_vel.setZero();
            return robot_action;
        }
        warned_missing_perception_ = false;

        Vec3f base_omgea = ro.base_omega * omega_scale_;
        Vec3f projected_gravity = ro.base_rot_mat.inverse() * gravity_direction;
        Vec3f command = Vec3f(uc.forward_vel_scale, uc.side_vel_scale, uc.turnning_vel_scale);

        for (int i = 0; i < action_dim; ++i){
            joint_pos_rl(i) = ro.joint_pos(robot2policy_idx[i]);
            joint_vel_rl(i) = ro.joint_vel(robot2policy_idx[i]) * dof_vel_scale_;
        }
        joint_pos_rl.segment(12, 4).setZero();

        joint_pos_rl -= dof_default_eigen_policy;
        int offset = 0;
        current_observation_.segment(offset, 3) = base_omgea; offset += 3;
        current_observation_.segment(offset, 3) = projected_gravity; offset += 3;
        current_observation_.segment(offset, 3) = command; offset += 3;
        current_observation_.segment(offset, action_dim) = joint_pos_rl; offset += action_dim;
        current_observation_.segment(offset, action_dim) = joint_vel_rl; offset += action_dim;
        current_observation_.segment(offset, action_dim) = last_action_eigen; offset += action_dim;
        if (observation_mode_ == ObservationMode::kDepthDualInput) {
            current_observation_(offset++) = (*selected_depth_observation)(depth_tensor_dim);
            current_observation_(offset++) = (*selected_depth_observation)(depth_tensor_dim + 1);
            depth_input_ = selected_depth_observation->head(depth_tensor_dim);
        } else if (observation_mode_ == ObservationMode::kV11TwoRiser) {
            VecXf v11_feature = ro.v11_observation;
            // V11 training gates the two-riser actor channel unless commanded
            // forward velocity exceeds 0.05 m/s. Keep deployment ABI-equivalent.
            if (command(0) <= 0.05f) {
                v11_feature.setZero();
            }
            current_observation_.segment(offset, v11_feature_dim) = v11_feature;
            offset += v11_feature_dim;
        } else if (lidar_observation_dim_ > 0) {
            current_observation_.segment(offset, lidar_observation_dim_) = ro.lidar_observation;
            offset += lidar_observation_dim_;
        }
        if (offset != observation_dim_) {
            throw std::runtime_error("Internal M20 observation layout does not match the ONNX input");
        }

        if (!current_observation_.allFinite()) {
            if (!warned_invalid_observation_) {
                std::cerr << "Policy observation contains NaN/Inf; holding the default pose and stopping wheels."
                          << std::endl;
                warned_invalid_observation_ = true;
            }
            robot_action.goal_joint_pos = dof_default_eigen_robot;
            robot_action.goal_joint_vel.setZero();
            return robot_action;
        }
        warned_invalid_observation_ = false;

        current_action_eigen = Onnx_infer(current_observation_);
        if (!current_action_eigen.allFinite()) {
            if (!warned_invalid_action_) {
                std::cerr << "Policy returned NaN/Inf; holding the default pose and stopping wheels."
                          << std::endl;
                warned_invalid_action_ = true;
            }
            robot_action.goal_joint_pos = dof_default_eigen_robot;
            robot_action.goal_joint_vel.setZero();
            return robot_action;
        }
        warned_invalid_action_ = false;

        if (zero_command_wheel_stop_ && command.norm() < zero_command_wheel_stop_threshold) {
            current_action_eigen.segment(12, 4).setZero();
        }
        last_action_eigen = current_action_eigen;

        
        for (int i = 0; i < action_dim; ++i){
            tmp_action_eigen(i) = current_action_eigen(policy2robot_idx[i]);
            tmp_action_eigen(i) *= action_scale_robot[i];
        }
        tmp_action_eigen += dof_default_eigen_robot;
        
        for (int i = 0; i < 4; ++i){
            robot_action.goal_joint_pos.segment(i*4, 3) = tmp_action_eigen.segment(i*4, 3);
            robot_action.goal_joint_vel(i*4+3) = tmp_action_eigen(i*4+3);
        }

        if (debug_policy_io_ && debug_policy_io_count_ < 20) {
            std::cout << "[M20_POLICY_DEBUG] step=" << debug_policy_io_count_
                      << " cmd=" << command.transpose()
                      << " gravity=" << projected_gravity.transpose()
                      << " joint_pos_rel=" << joint_pos_rl.transpose()
                      << " joint_vel_scaled=" << joint_vel_rl.transpose()
                      << " raw_action=" << current_action_eigen.transpose()
                      << " robot_cmd=" << tmp_action_eigen.transpose()
                      << std::endl;
            ++debug_policy_io_count_;
        }

        
        ++run_cnt_;
        ++time_step;
        return robot_action;
    }

    void setDefaultJointPos(const VecXf& pos){
        dof_pos_default_.setZero(motor_num); 
        for(int i=0;i<motor_num;++i) {
            dof_pos_default_(i) = pos(i);
        }
    }

    double getCurrentTime() {
        clock_gettime(1, &system_time);
        return system_time.tv_sec + system_time.tv_nsec / 1e9;
    }
};
