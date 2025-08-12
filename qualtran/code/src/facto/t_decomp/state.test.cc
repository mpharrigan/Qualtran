#include "state.h"

#include "gtest/gtest.h"

using namespace decomp;

TEST(state, x) {
    ASSERT_EQ(State::seed0().after_x().after_x().amp0, State::seed0().amp0);
}
