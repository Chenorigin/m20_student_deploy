#pragma once
#include "common_types.h"
#include "policy_fault_recovery.hpp"
namespace interface {
class UserCommandInterface {
    types::UserCommand command_;
public:
    m20::PolicyFaultRecovery policy_recovery;
    types::UserCommand* GetUserCommand(){return &command_;}
};
}
