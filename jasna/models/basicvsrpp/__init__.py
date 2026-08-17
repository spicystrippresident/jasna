# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from jasna.models.basicvsrpp.mmengine_compat import prepare_mmengine_for_windows_rocm


# MMEngine imports distributed-only symbols before it exposes the model/config
# helpers used below.  Windows ROCm inference is single-process and its torch
# wheel omits those symbols, so prepare that dependency boundary first.
prepare_mmengine_for_windows_rocm()


def register_all_modules():
    from jasna.models.basicvsrpp.mmagic import register_all_modules
    register_all_modules()
    from jasna.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGanNet, BasicVSRPlusPlusGan
