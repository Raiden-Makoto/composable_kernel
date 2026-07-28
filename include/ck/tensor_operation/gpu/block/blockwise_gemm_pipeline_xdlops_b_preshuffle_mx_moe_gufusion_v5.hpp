// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/block/blockwise_gemm_mx_pipeline_xdlops_base.hpp"

namespace ck {

// Naive pipeline with lowest resource request per WGP
// GlobalPrefetchStages: 2
// LocalPreFillStages: 1
// LocalPreFetchStages: 1
// LocalSharedMemoryBuffer: 1

template <BlockGemmPipelineScheduler BlkGemmPipelineVer,
          index_t ThreadBlockSize,
          index_t ScaleBlockSize,
          typename ADataType,
          typename AScaleDataType,
          typename BDataType,
          typename BScaleDataType,
          typename ATileDesc,
          typename BTileDesc,
          typename AMmaTileDesc,
          typename BMmaTileDesc,
          index_t ABlockTransferSrcScalarPerVector,
          index_t BBlockTransferSrcScalarPerVector,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat, // MXdlPerWave
          index_t NRepeat, // NXdlPerWave
          index_t KPack>
struct BlockwiseGemmXdlops_pipeline_bpreshuffle_mx_moe_gufusion_v5
{
};

template <index_t ThreadBlockSize,
          index_t ScaleBlockSize,
          typename ADataType,
          typename AScaleDataType,
          typename BDataType,
          typename BScaleDataType,
          typename ATileDesc,
          typename BTileDesc,
          typename AMmaTileDesc,
          typename BMmaTileDesc,
          index_t ABlockTransferSrcScalarPerVector,
          index_t BBlockTransferSrcScalarPerVector,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat, // MXdlPerWave
          index_t NRepeat, // NXdlPerWave
          index_t KPack>
struct BlockwiseGemmXdlops_pipeline_bpreshuffle_mx_moe_gufusion_v5<
    BlockGemmPipelineScheduler::Intrawave,
    ThreadBlockSize,
    ScaleBlockSize,
    ADataType,
    AScaleDataType,
    BDataType,
    BScaleDataType,
    ATileDesc,
    BTileDesc,
    AMmaTileDesc,
    BMmaTileDesc,
    ABlockTransferSrcScalarPerVector,
    BBlockTransferSrcScalarPerVector,
    MPerBlock,
    NPerBlock,
    KPerBlock,
    MPerXDL,
    NPerXDL,
    MRepeat,
    NRepeat,
    KPack> : BlockwiseGemmXdlops_mx_pipeline_base<ThreadBlockSize,
                                                  ADataType,
                                                  BDataType,
                                                  ATileDesc,
                                                  BTileDesc,
                                                  AMmaTileDesc,
                                                  BMmaTileDesc,
                                                  ABlockTransferSrcScalarPerVector,
                                                  BBlockTransferSrcScalarPerVector,
                                                  MPerBlock,
                                                  NPerBlock,
                                                  KPerBlock,
                                                  MPerXDL,
                                                  NPerXDL,
                                                  MRepeat,
                                                  NRepeat,
                                                  KPack>

{

    using Base = BlockwiseGemmXdlops_mx_pipeline_base<ThreadBlockSize,
                                                      ADataType,
                                                      BDataType,
                                                      ATileDesc,
                                                      BTileDesc,
                                                      AMmaTileDesc,
                                                      BMmaTileDesc,
                                                      ABlockTransferSrcScalarPerVector,
                                                      BBlockTransferSrcScalarPerVector,
                                                      MPerBlock,
                                                      NPerBlock,
                                                      KPerBlock,
                                                      MPerXDL,
                                                      NPerXDL,
                                                      MRepeat,
                                                      NRepeat,
                                                      KPack>;
    using Base::A_K1;
    using Base::I0;
    using Base::I1;
    using Base::KRepeat;
    using Base::MWaves;
    using Base::NWaves;
    using Base::WaveSize;
    using Base::xdlops_gemm;
    using typename Base::HotLoopInstList;

    using Base::CalculateCThreadOriginDataIndex;
    using Base::GetCBlockDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4;
    using Base::GetCThreadBuffer;
    using Base::GetCThreadDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4;
    using Base::GetWaveIdx;
    using Base::MakeCGridDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2;

    using Base::a_block_desc_m0_m1_m2_m3_k;
    using Base::b_block_desc_n0_n1_n2_n3_k;

    using Base::AMmaKStride;
    using Base::APackedSize;
    using Base::BMmaKStride;
    using Base::BPackedSize;
    using Base::KThreadChunk;

    using Base::KXdlPack;
    using Base::MXdlPack;
    using Base::NXdlPack;

    using AccType      = typename Base::AccType;
    using Tuple5       = typename Base::Tuple5;
    using ComputeTypeA = typename Base::ComputeTypeA;
    using ComputeTypeB = typename Base::ComputeTypeB;

    static constexpr index_t PrefetchStages        = 2;
    static constexpr index_t LocalPrefetchStages   = 2;
    static constexpr index_t PrefillStages         = 1;
    static constexpr index_t GlobalBufferNum       = 1;
    static constexpr index_t HotloopLocalBufSwitch = MRepeat % 2 == 0 ? 0 : 1;

    static constexpr auto num_buffer_load_a_scale = MRepeat / MXdlPack * KRepeat / KXdlPack;
    static constexpr auto num_buffer_load_b_scale = NRepeat / NXdlPack * KRepeat / KXdlPack * 2;
    static constexpr auto async_vmcnt = num_buffer_load_a_scale + num_buffer_load_b_scale +
                                        HotLoopInstList::B_Buffer_Load_Inst_Num * 2;
    static constexpr auto async_vmcnt_encoding = 3952 + async_vmcnt % 16 + async_vmcnt / 16 * 16384;

    // One K tile is staged through LDS as a whole but consumed as KStageRepeat register stages,
    // so only one stage of A/B/scale fragments is live in registers at a time.
    static constexpr index_t KStageRepeat = 2;
    static constexpr index_t KStage      = KRepeat / KStageRepeat;

    // Only one stage worth of B and scale loads is in flight during the prologue.
    static constexpr auto stage_vmcnt = async_vmcnt / KStageRepeat;
    static constexpr auto stage_vmcnt_encoding = 3952 + stage_vmcnt % 16 + stage_vmcnt / 16 * 16384;

    static constexpr auto ScalesPerKBlockSize =
        KPerBlock / ScaleBlockSize; // How many mx-vectors per K block

    //> How many mx-vectors in each row/col is processed in one call to xdlops_gemm.Run()
    static constexpr auto ScalesPerXdlopsRun =
        (APackedSize * KPack * xdlops_gemm.K0PerXdlops) / ScaleBlockSize;

    //> How many scales a thread must read to accommodate one call to xdlops_gemm.Run()
    static constexpr auto ScalesPerXdlopsRunPerThread =
        ScalesPerXdlopsRun / xdlops_gemm.mfma_instr.num_input_blks;

    using mx_scale_t                        = e8m0_bexp_t;
    static constexpr auto scale_pack_size_a = sizeof(AScaleDataType) / sizeof(mx_scale_t);
    static constexpr auto scale_pack_size_b = sizeof(BScaleDataType) / sizeof(mx_scale_t);
    static_assert(KXdlPack * MXdlPack % scale_pack_size_a == 0,
                  "A scale pack data type too large!");
    static_assert(KXdlPack * NXdlPack % scale_pack_size_b == 0,
                  "B scale pack data type too large!");
    static constexpr auto a_scale_thread_vec_size = KXdlPack * MXdlPack / scale_pack_size_a;
    static constexpr auto b_scale_thread_vec_size = KXdlPack * NXdlPack / scale_pack_size_b;

    __host__ static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    __host__ static constexpr TailNumber BlockLoopTailNum(index_t num_loop)
    {
        return num_loop % 2 == 0 ? TailNumber::Even : TailNumber::Odd;
    }

    __device__ static constexpr auto HotLoopScheduler()
    {
        // A/B split schedule
        // compiler is likely to use ds_read2 when instruction width smaller than 16bytes
        constexpr auto num_ds_read_inst_a =
            HotLoopInstList::A_LDS_Read_Width * sizeof(ADataType) == 16
                ? HotLoopInstList::A_LDS_Read_Inst_Num
                : HotLoopInstList::A_LDS_Read_Inst_Num / 2;

        constexpr auto num_buffer_load_inst_a = HotLoopInstList::A_Buffer_Load_Inst_Num;
        constexpr auto num_buffer_load_inst_b = HotLoopInstList::B_Buffer_Load_Inst_Num * 2;
        constexpr auto num_buffer_load_stage1 =
            num_buffer_load_inst_b + num_buffer_load_a_scale + num_buffer_load_b_scale;

        constexpr auto num_buffer_load_stage2 = num_buffer_load_inst_a;

        constexpr auto num_mfma_inst = HotLoopInstList::C_MFMA_Inst_Num * APackedSize * 2;
        constexpr auto mfma_cycle    = HotLoopInstList::C_MFMA_Inst_Cycle;

        constexpr auto ds_read_a_issue_cycle =
            HotLoopInstList::A_LDS_Read_Width * sizeof(ADataType) == 16 ? 8 : 4;
        constexpr auto ds_read_a_mfma_rate =
            math::integer_divide_ceil(mfma_cycle - 8, 2 * ds_read_a_issue_cycle);

        // constexpr auto num_dsread_a_mfma =
        //     (num_ds_read_inst_a + ds_read_a_mfma_rate - 1) / ds_read_a_mfma_rate;

        constexpr auto num_total_stages = MRepeat;

        // Group num_mfma_perstage num_ds_read_a_perstage
        // since we want to reuse a local register buffer
        constexpr auto num_mfma_perstage      = num_mfma_inst / num_total_stages;
        constexpr auto num_ds_read_a_perstage = num_ds_read_inst_a / num_total_stages;

        constexpr auto num_ds_read_a_mfma_perstage =
            math::integer_divide_ceil(num_ds_read_a_perstage, ds_read_a_mfma_rate);

        constexpr auto num_ds_read_a_prefetch_stages = 2;

        constexpr auto buffer_load_perstage_more =
            math::integer_divide_ceil((num_buffer_load_stage1), (num_total_stages - 2));
        constexpr auto buffer_load_perstage_less =
            math::integer_divide_floor((num_buffer_load_stage1), (num_total_stages - 2));
        constexpr auto buffer_load_perstage_stage2 =
            math::integer_divide_floor((num_buffer_load_stage2), 2);

        constexpr auto buffer_load_stages_more =
            num_buffer_load_stage1 -
            math::integer_divide_floor(num_buffer_load_stage1, (num_total_stages - 2)) *
                ((num_total_stages - 2));

        constexpr auto buffer_load_issue_point_interval_more =
            num_mfma_perstage / buffer_load_perstage_more;
        constexpr auto buffer_load_issue_point_interval_less =
            num_mfma_perstage / buffer_load_perstage_less;
        constexpr auto buffer_load_issue_point_interval_stage2 =
            num_mfma_perstage / buffer_load_perstage_stage2;

        // Stage 1
        // global read more
        static_ford<Sequence<buffer_load_stages_more, num_mfma_perstage>>{}([&](auto ii) {
            constexpr auto imfma = Number<ii[Number<1>{}]>{};
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA

            if constexpr(imfma % buffer_load_issue_point_interval_more == 0)
            {
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
            }

            if constexpr(imfma >= (num_mfma_perstage - num_ds_read_a_mfma_perstage))
            {
                __builtin_amdgcn_sched_group_barrier(0x100, ds_read_a_mfma_rate, 0); // DS read
            }
        });

        // global read less
        static_ford<
            Sequence<(num_total_stages - 2 - buffer_load_stages_more), num_mfma_perstage>>{}(
            [&](auto ii) {
                constexpr auto imfma = Number<ii[Number<1>{}]>{};
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                if constexpr(imfma % buffer_load_issue_point_interval_less == 0)
                {
                    __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                }
                if constexpr(imfma >= (num_mfma_perstage - num_ds_read_a_mfma_perstage))
                {
                    __builtin_amdgcn_sched_group_barrier(0x100, ds_read_a_mfma_rate, 0); // DS read
                }
            });

        // Stage 2, Sync
        // lds synchronization, prefetch next loop local A
        static_ford<Sequence<num_ds_read_a_prefetch_stages, num_mfma_perstage>>{}([&](auto ii) {
            constexpr auto imfma = Number<ii[Number<1>{}]>{};
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            if constexpr(imfma % buffer_load_issue_point_interval_stage2 == 0)
            {
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
            }
            if constexpr(imfma >= (num_mfma_perstage - num_ds_read_a_mfma_perstage))
            {
                __builtin_amdgcn_sched_group_barrier(0x100, ds_read_a_mfma_rate, 0); // DS read
            }
        });
    }

    template <bool HasMainLoop,
              TailNumber TailNum,
              typename AGridDesc,
              typename ABlockDesc,
              typename ABlockTransfer,
              typename AGridBuffer,
              typename ABlockBuffer,
              typename ABlockTransferStep,
              typename BGridDesc,
              typename BBlockDesc,
              typename BBlockTransfer,
              typename BGridBuffer,
              typename BBlockBuffer,
              typename BBlockTransferStep,
              typename CThreadBuffer,
              typename AScaleGridBuffer,
              typename AScaleGridDesc,
              typename AScaleThreadTransfer,
              typename BScaleGridBuffer,
              typename BScaleGridDesc,
              typename BScaleThreadTransfer>
    __device__ void Run(
        // ABlockCopy
        const AGridDesc& a_grid_desc,
        const ABlockDesc& a_block_desc,
        ABlockTransfer& a_blockwise_copy,
        const AGridBuffer& a_grid_buf,
        ABlockBuffer& a_block_bufs,
        const ABlockTransferStep& a_block_copy_step,
        // B Gate and Up
        const BGridDesc& b_grid_desc,
        const BBlockDesc& b_block_desc,
        BBlockTransfer& b_blockwise_copy,
        BBlockTransfer& b_blockwise_copy_up,
        const BGridBuffer& b_grid_buf,
        const BGridBuffer& b_grid_buf_up,
        BBlockBuffer& b_block_bufs,
        const BBlockTransferStep& b_block_copy_step,
        // CThread
        CThreadBuffer& c_thread_buf,
        CThreadBuffer& c_thread_buf_up,
        // A and B scales
        const AScaleGridDesc& a_scale_grid_desc,
        AScaleThreadTransfer& a_scale_thread_copy,
        const AScaleGridBuffer& a_scale_grid_buf,
        // Gate and Up scale
        const BScaleGridDesc& b_scale_grid_desc,
        BScaleThreadTransfer& b_scale_thread_copy,
        BScaleThreadTransfer& b_scale_thread_copy_up,
        const BScaleGridBuffer& b_scale_grid_buf,
        const BScaleGridBuffer& b_scale_grid_buf_up,
        index_t num_loop) const
    {
        ignore = b_block_bufs;
        ignore = b_blockwise_copy_up;
        ignore = b_block_copy_step;

        static_assert(KRepeat % KStageRepeat == 0 && KStage % KXdlPack == 0,
                      "gufusion v5 needs a K tile that splits into whole K stages");

        StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                  ComputeTypeA,
                                  a_thread_desc_.GetElementSpaceSize() / KPack,
                                  KPack,
                                  true>
            a_thread_buf;
        StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                  ComputeTypeB,
                                  b_thread_desc_stage_.GetElementSpaceSize() / KPack,
                                  KPack,
                                  true>
            b_thread_buf;
        StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                  ComputeTypeB,
                                  b_thread_desc_stage_.GetElementSpaceSize() / KPack,
                                  KPack,
                                  true>
            b_thread_buf_up;

        StaticallyIndexedArray<decltype(b_thread_buf), Number<2>{}> b_thread_bufs;
        StaticallyIndexedArray<decltype(b_thread_buf_up), Number<2>{}> b_thread_bufs_up;

        auto b_block_offset = b_blockwise_copy.GetSrcOffset();
        const auto b_block_n_stride =
            b_grid_desc.CalculateOffset(make_multi_index(I1, I0, I0, I0, I0)) -
            b_grid_desc.CalculateOffset(make_multi_index(I0, I0, I0, I0, I0));
        const auto b_block_xdl_stride =
            b_grid_desc.CalculateOffset(make_multi_index(I0, I0, I1, I0, I0)) -
            b_grid_desc.CalculateOffset(make_multi_index(I0, I0, I0, I0, I0));
        const auto b_block_k_stride =
            b_grid_desc.CalculateOffset(make_multi_index(I0, I0, I0, I1, I0)) -
            b_grid_desc.CalculateOffset(make_multi_index(I0, I0, I0, I0, I0));

        using BBlockLoad =
            typename vector_type<BDataType, BBlockTransferSrcScalarPerVector>::type;

        auto load_b_stage = [&](auto stage_buf) {
            static_for<0, NRepeat / NXdlPack, 1>{}([&](auto n0) {
                static_for<0, NXdlPack, 1>{}([&](auto inxdl) {
                    static_for<0, KStage, 1>{}([&](auto k0) {
                        const auto src_offset = b_block_offset +
                                                n0 * b_block_n_stride +
                                                inxdl * b_block_xdl_stride +
                                                k0 * b_block_k_stride;
                        const auto gate =
                            b_grid_buf.template Get<BBlockLoad>(src_offset, true);
                        const auto up =
                            b_grid_buf_up.template Get<BBlockLoad>(src_offset, true);
                        constexpr auto dst_offset = b_thread_desc_stage_.CalculateOffset(
                            make_tuple(n0, I0, inxdl, k0, I0));

                        static_for<0, BBlockTransferSrcScalarPerVector, 1>{}([&](auto s) {
                            b_thread_bufs(stage_buf)(Number<dst_offset + s>{}) =
                                type_convert<BDataType>(gate.template AsType<BDataType>()[s]);
                            b_thread_bufs_up(stage_buf)(Number<dst_offset + s>{}) =
                                type_convert<BDataType>(up.template AsType<BDataType>()[s]);
                        });
                    });
                });
            });

            b_block_offset += KStage * b_block_k_stride;
        };

        auto a_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, AScaleDataType>(
            a_scale_thread_desc_stage_.GetElementSpaceSize());

        auto b_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, BScaleDataType>(
            b_scale_thread_desc_stage_.GetElementSpaceSize());
        auto b_scale_thread_buf_up = make_static_buffer<AddressSpaceEnum::Vgpr, BScaleDataType>(
            b_scale_thread_desc_stage_.GetElementSpaceSize());

        StaticallyIndexedArray<decltype(a_scale_thread_buf), Number<2>{}> a_scale_thread_bufs;
        StaticallyIndexedArray<decltype(b_scale_thread_buf), Number<2>{}> b_scale_thread_bufs;
        StaticallyIndexedArray<decltype(b_scale_thread_buf_up), Number<2>{}> b_scale_thread_bufs_up;

        auto a_scale_block_offset = a_scale_thread_copy.GetSrcOffset();
        auto b_scale_block_offset = b_scale_thread_copy.GetSrcOffset();

        const auto a_scale_m_stride =
            a_scale_grid_desc.CalculateOffset(make_multi_index(I1, I0, I0)) -
            a_scale_grid_desc.CalculateOffset(make_multi_index(I0, I0, I0));
        const auto a_scale_k_stride =
            a_scale_grid_desc.CalculateOffset(make_multi_index(I0, I1, I0)) -
            a_scale_grid_desc.CalculateOffset(make_multi_index(I0, I0, I0));
        const auto b_scale_n_stride =
            b_scale_grid_desc.CalculateOffset(make_multi_index(I1, I0, I0)) -
            b_scale_grid_desc.CalculateOffset(make_multi_index(I0, I0, I0));
        const auto b_scale_k_stride =
            b_scale_grid_desc.CalculateOffset(make_multi_index(I0, I1, I0)) -
            b_scale_grid_desc.CalculateOffset(make_multi_index(I0, I0, I0));

        using AScaleLoad = typename vector_type<AScaleDataType, a_scale_thread_vec_size>::type;
        using BScaleLoad = typename vector_type<BScaleDataType, b_scale_thread_vec_size>::type;

        auto load_scale_stage = [&](auto stage_buf) {
            static_for<0, MRepeat / MXdlPack, 1>{}([&](auto m0) {
                static_for<0, KStage / KXdlPack, 1>{}([&](auto k0) {
                    const auto src_offset = a_scale_block_offset +
                                            m0 * MWaves * a_scale_m_stride +
                                            k0 * a_scale_k_stride;
                    const auto scale =
                        a_scale_grid_buf.template Get<AScaleLoad>(src_offset, true);
                    constexpr auto dst_offset = a_scale_thread_desc_stage_.CalculateOffset(
                        make_tuple(m0, k0, I0));

                    static_for<0, a_scale_thread_vec_size, 1>{}([&](auto s) {
                        a_scale_thread_bufs(stage_buf)(Number<dst_offset + s>{}) =
                            type_convert<AScaleDataType>(
                                scale.template AsType<AScaleDataType>()[s]);
                    });
                });
            });

            static_for<0, NRepeat / NXdlPack, 1>{}([&](auto n0) {
                static_for<0, KStage / KXdlPack, 1>{}([&](auto k0) {
                    const auto src_offset = b_scale_block_offset +
                                            n0 * NWaves * b_scale_n_stride +
                                            k0 * b_scale_k_stride;
                    const auto gate_scale =
                        b_scale_grid_buf.template Get<BScaleLoad>(src_offset, true);
                    const auto up_scale =
                        b_scale_grid_buf_up.template Get<BScaleLoad>(src_offset, true);
                    constexpr auto dst_offset = b_scale_thread_desc_stage_.CalculateOffset(
                        make_tuple(n0, k0, I0));

                    static_for<0, b_scale_thread_vec_size, 1>{}([&](auto s) {
                        b_scale_thread_bufs(stage_buf)(Number<dst_offset + s>{}) =
                            type_convert<BScaleDataType>(
                                gate_scale.template AsType<BScaleDataType>()[s]);
                        b_scale_thread_bufs_up(stage_buf)(Number<dst_offset + s>{}) =
                            type_convert<BScaleDataType>(
                                up_scale.template AsType<BScaleDataType>()[s]);
                    });
                });
            });

            a_scale_block_offset += (KStage / KXdlPack) * a_scale_k_stride;
            b_scale_block_offset += (KStage / KXdlPack) * b_scale_k_stride;
        };

        // Read one K stage of A fragments for one M slot out of the resident K tile in LDS.
        auto read_a_stage = [&](auto m_major, auto im_minor, auto stage, const auto& lds_buf) {
            static_for<0, KStage, 1>{}([&](auto k) {
                constexpr auto k_global = Number<stage.value * KStage + k.value>{};
                constexpr auto k_step   = k_global * xdlops_gemm.KPerXdlops / APackedSize *
                                        (APackedSize * KPack / xdlops_gemm.K1PerXdlops);
                static_for<0, xdlops_gemm.K1PerXdlops / (APackedSize * KThreadChunk), 1>{}(
                    [&](auto chunk) {
                        constexpr auto a_k_step_chunk =
                            k_step + chunk * KThreadChunk * xdlops_gemm.mfma_instr.num_input_blks;
                        a_thread_copy_.Run(
                            a_block_desc_m0_m1_m2_m3_k,
                            make_tuple(m_major, I0, im_minor, I0, Number<a_k_step_chunk>{}),
                            lds_buf,
                            a_thread_desc_,
                            make_tuple(I0, I0, im_minor, k, Number<chunk * KThreadChunk>{}),
                            a_thread_buf);
                    });
            });
        };

        // All MFMAs of one M slot against the K stage currently held in registers.
        auto compute_stage_m = [&](auto stage_buf, auto m0) {
            constexpr auto im_major = m0 / MXdlPack;
            constexpr auto im_minor = m0 % MXdlPack;
            static_ford<Sequence<KStage, NRepeat>>{}([&](auto kn) {
                constexpr auto k0       = Number<kn[Number<0>{}]>{};
                constexpr auto n0       = Number<kn[Number<1>{}]>{};
                constexpr auto ik_major = k0 / KXdlPack;
                constexpr auto ik_minor = k0 % KXdlPack;
                constexpr auto in_major = n0 / NXdlPack;
                constexpr auto in_minor = n0 % NXdlPack;

                constexpr index_t a_scale_offset = a_scale_thread_desc_stage_.CalculateOffset(
                    make_tuple(im_major, ik_major, I0));
                constexpr index_t b_scale_offset = b_scale_thread_desc_stage_.CalculateOffset(
                    make_tuple(in_major, ik_major, I0));

                static_assert(0 < ScalesPerXdlopsRunPerThread,
                              "Must have at least one scale per Xdlops "
                              "per Thread.");

                vector_type<AScaleDataType, a_scale_thread_vec_size> a_scale_thread_vec;
                vector_type<BScaleDataType, b_scale_thread_vec_size> b_scale_thread_vec;
                vector_type<BScaleDataType, b_scale_thread_vec_size> b_scale_thread_vec_up;

                // Pack scale_thread_buf into scale_thread_vec
                static_for<0, a_scale_thread_vec_size, 1>{}([&](auto s) {
                    a_scale_thread_vec.template AsType<AScaleDataType>()(s) =
                        a_scale_thread_bufs(stage_buf)[Number<a_scale_offset + s>{}];
                });
                // B Gate scale
                static_for<0, b_scale_thread_vec_size, 1>{}([&](auto s) {
                    b_scale_thread_vec.template AsType<BScaleDataType>()(s) =
                        b_scale_thread_bufs(stage_buf)[Number<b_scale_offset + s>{}];
                });
                // B Up scale
                static_for<0, b_scale_thread_vec_size, 1>{}([&](auto s) {
                    b_scale_thread_vec_up.template AsType<BScaleDataType>()(s) =
                        b_scale_thread_bufs_up(stage_buf)[Number<b_scale_offset + s>{}];
                });

                constexpr auto a_thread_offset =
                    a_thread_desc_.CalculateOffset(make_tuple(I0, I0, im_minor, k0, I0));
                constexpr auto b_thread_offset = b_thread_desc_stage_.CalculateOffset(
                    make_tuple(in_major, I0, in_minor, k0, I0));
                const auto& a_thread_vec =
                    a_thread_buf.GetVectorTypeReference(Number<a_thread_offset>{});
                const auto& b_thread_vec =
                    b_thread_bufs[stage_buf].GetVectorTypeReference(Number<b_thread_offset>{});
                const auto& b_thread_vec_up =
                    b_thread_bufs_up[stage_buf].GetVectorTypeReference(Number<b_thread_offset>{});

                using mfma_input_type_a =
                    typename vector_type<ComputeTypeA,
                                         xdlops_gemm.K1PerXdlops / APackedSize>::type;
                using mfma_input_type_b =
                    typename vector_type<ComputeTypeB,
                                         xdlops_gemm.K1PerXdlops / BPackedSize>::type;

                using mfma_scale_input_type_a =
                    typename vector_type<AScaleDataType, a_scale_thread_vec_size>::type;
                using mfma_scale_input_type_b =
                    typename vector_type<BScaleDataType, b_scale_thread_vec_size>::type;

                constexpr index_t c_offset = c_thread_desc_.CalculateOffset(
                    make_tuple(im_major, in_major, im_minor, in_minor, 0));

                // MFMA accumulation A * Gate
                xdlops_gemm.template Run<ik_minor * MXdlPack + im_minor,
                                         ik_minor * NXdlPack + in_minor>(
                    a_thread_vec.template AsType<mfma_input_type_a>(),
                    a_scale_thread_vec.template AsType<mfma_scale_input_type_a>(),
                    b_thread_vec.template AsType<mfma_input_type_b>(),
                    b_scale_thread_vec.template AsType<mfma_scale_input_type_b>(),
                    c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));

                // MFMA accumulation A * Up
                xdlops_gemm.template Run<ik_minor * MXdlPack + im_minor,
                                         ik_minor * NXdlPack + in_minor>(
                    a_thread_vec.template AsType<mfma_input_type_a>(),
                    a_scale_thread_vec.template AsType<mfma_scale_input_type_a>(),
                    b_thread_vec_up.template AsType<mfma_input_type_b>(),
                    b_scale_thread_vec_up.template AsType<mfma_scale_input_type_b>(),
                    c_thread_buf_up.GetVectorTypeReference(Number<c_offset>{}));

                if constexpr(MPerBlock == 64 && k0.value == 0 && n0.value == 0)
                {
                    __builtin_amdgcn_sched_barrier(0);
                    __builtin_amdgcn_s_setprio(1);
                    __builtin_amdgcn_sched_barrier(0);
                }
            });

            if constexpr(MPerBlock == 64)
            {
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_sched_barrier(0);
            }
        };

        constexpr index_t SwitchM = MRepeat - LocalPrefetchStages;

        // One K tile: KStageRepeat register stages over the same resident LDS tile, no barrier
        // between stages. GlobalCopy/LdsSync/LoadAfterLast/PrefetchNextTile are Number<0/1> flags.
        auto tile_pass = [&](auto comp_lds,
                             auto mem_lds,
                             auto do_global_copy,
                             auto do_lds_sync,
                             auto load_after_last,
                             auto prefetch_next_tile) {
            static_for<0, KStageRepeat, 1>{}([&](auto stage) {
                constexpr auto cur_buf     = Number<stage.value % 2>{};
                constexpr auto nxt_buf     = Number<(stage.value + 1) % 2>{};
                constexpr bool is_last     = stage.value == (KStageRepeat - 1);

                if constexpr(!is_last || load_after_last.value != 0)
                {
                    load_b_stage(nxt_buf);
                    load_scale_stage(nxt_buf);
                }

                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    compute_stage_m(cur_buf, m0);

                    if constexpr(is_last && m0.value == SwitchM)
                    {
                        if constexpr(do_lds_sync.value != 0)
                        {
                            __builtin_amdgcn_s_waitcnt(async_vmcnt_encoding);
                            block_sync_lds();
                        }
                        if constexpr(do_global_copy.value != 0)
                        {
                            a_blockwise_copy.Run(
                                a_grid_desc, a_grid_buf, a_block_desc, a_block_bufs(comp_lds));
                            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                        }
                    }

                    constexpr auto m_major = Number<((m0 + LocalPrefetchStages) / MXdlPack) %
                                                    (MRepeat / MXdlPack)>{};
                    constexpr auto im_minor = Number<m0 % MXdlPack>{};

                    if constexpr(m0.value < SwitchM)
                    {
                        read_a_stage(m_major, im_minor, stage, a_block_bufs(comp_lds));
                    }
                    else if constexpr(!is_last)
                    {
                        read_a_stage(
                            m_major, im_minor, Number<stage.value + 1>{}, a_block_bufs(comp_lds));
                    }
                    else if constexpr(prefetch_next_tile.value != 0)
                    {
                        read_a_stage(m_major, im_minor, Number<0>{}, a_block_bufs(mem_lds));
                    }
                });
            });
        };

        // Global prefetch 1
        a_blockwise_copy.Run(a_grid_desc, a_grid_buf, a_block_desc, a_block_bufs(I0));
        load_b_stage(I0);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);

        load_scale_stage(I0);

        // Local prefetch 1, sync the async load
        __builtin_amdgcn_s_waitcnt(stage_vmcnt_encoding);
        block_sync_lds();
        static_for<0, LocalPrefetchStages, 1>{}([&](auto m0) {
            read_a_stage(Number<(m0 / MXdlPack) % (MRepeat / MXdlPack)>{},
                         Number<m0 % MXdlPack>{},
                         Number<0>{},
                         a_block_bufs(I0));
        });

        // Global prefetch 2
        a_blockwise_copy.Run(a_grid_desc, a_grid_buf, a_block_desc, a_block_bufs(I1));
        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();
        __builtin_amdgcn_sched_barrier(0);

        // main body
        if constexpr(HasMainLoop)
        {
            // loop over k with the step KPerBlock
            index_t i = 0;
            do
            {
                tile_pass(I0, I1, Number<1>{}, Number<1>{}, Number<1>{}, Number<1>{});
                __builtin_amdgcn_sched_barrier(0);
                tile_pass(I1, I0, Number<1>{}, Number<1>{}, Number<1>{}, Number<1>{});
                __builtin_amdgcn_sched_barrier(0);

                i += 2;
            } while(i < (num_loop - 2));
        }

        // tail
        if constexpr(TailNum == TailNumber::Even)
        {
            tile_pass(I0, I1, Number<0>{}, Number<1>{}, Number<1>{}, Number<1>{});
            __builtin_amdgcn_sched_barrier(0);
            tile_pass(I1, I0, Number<0>{}, Number<0>{}, Number<0>{}, Number<0>{});
        }
        else if constexpr(TailNum == TailNumber::Odd)
        {
            tile_pass(I0, I1, Number<0>{}, Number<0>{}, Number<0>{}, Number<0>{});
        }
    }

    //  Length:  A[ARegBuf, MWave, MXdlPack, KStage, KPack]
    //  Order:     1        0      3         2       4
    static constexpr auto ARegBuf        = 2;
    static constexpr auto a_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<ARegBuf>{}, I1, Number<MXdlPack>{}, Number<KStage>{}, Number<KPack>{}));

    static constexpr auto b_thread_desc_stage_ =
        make_naive_tensor_descriptor_packed(make_tuple(Number<NRepeat / NXdlPack>{},
                                                      I1,
                                                      Number<NXdlPack>{},
                                                      Number<KStage>{},
                                                      Number<KPack>{}));

    static constexpr auto a_scale_thread_desc_stage_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MRepeat / MXdlPack>{},
                   Number<KStage / KXdlPack>{},
                   Number<ScalesPerXdlopsRunPerThread * a_scale_thread_vec_size>{}));

    static constexpr auto b_scale_thread_desc_stage_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<NRepeat / NXdlPack>{},
                   Number<KStage / KXdlPack>{},
                   Number<ScalesPerXdlopsRunPerThread * b_scale_thread_vec_size>{}));

    using AThreadCopy = ThreadwiseTensorSliceTransfer_v4<ADataType,
                                                         ComputeTypeA,
                                                         decltype(a_block_desc_m0_m1_m2_m3_k),
                                                         decltype(a_thread_desc_),
                                                         Sequence<1, 1, 1, 1, KThreadChunk>,
                                                         Sequence<0, 1, 2, 3, 4>,
                                                         4,
                                                         A_K1,
                                                         A_K1>;
    AThreadCopy a_thread_copy_{Base::CalculateAThreadOriginDataIndex()};

    // TODO: make this field protected when a_scale_thread_copy_ is moved
    // here
    static constexpr auto a_scale_thread_desc = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MRepeat / MXdlPack>{},
                   Number<KRepeat / KXdlPack>{},
                   Number<ScalesPerXdlopsRunPerThread * a_scale_thread_vec_size>{}));

    // TODO: make this field protected when b_scale_thread_copy_ is moved
    // here
    static constexpr auto b_scale_thread_desc = make_naive_tensor_descriptor_packed(
        make_tuple(Number<NRepeat / NXdlPack>{},
                   Number<KRepeat / KXdlPack>{},
                   Number<ScalesPerXdlopsRunPerThread * b_scale_thread_vec_size>{}));

    protected:
    // using Base::a_thread_copy_;
    // using Base::a_thread_desc_;
    using Base::b_thread_copy_;
    using Base::b_thread_desc_;
    using Base::c_thread_desc_;
};

} // namespace ck
