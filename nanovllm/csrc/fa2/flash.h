#pragma once

#include "batch/batch_params.h"

// Backward-compatible aliases while csrc/fa2 migrates to layered layout.
using Qkv_params = QkvParams;
using Flash_fwd_params = BatchFwdParams;
