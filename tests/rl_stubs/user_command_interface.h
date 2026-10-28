#pragma once
#include "common_types.h"
namespace interface {
class UserCommandInterface {
    types::UserCommand command_;
public:
    types::UserCommand* GetUserCommand(){return &command_;}
};
}
