#include "MambaS6Plugin.h"
#include <cstring>
#include <iostream>
#include <vector>
#include <cassert>

// 宣告 Kernel Launcher
extern "C" void mambaS6KernelLauncher(
    const half *u, const half *delta, const half *A, const half *B, const half *C, const half *D,
    half *y,
    int B_size, int L_size, int D_model, int D_state,
    cudaStream_t stream);

namespace mamba
{

    // --- MambaS6Plugin Implementation ---

    MambaS6Plugin::MambaS6Plugin(const std::string &name, int d_state)
        : mName(name), mDState(d_state) {}

    MambaS6Plugin::MambaS6Plugin(const void *data, size_t length)
    {
        const char *d = static_cast<const char *>(data);
        const char *a = d;
        // 反序列化參數
        std::memcpy(&mDState, d, sizeof(mDState));
        d += sizeof(mDState);
        assert(d == a + length);
    }

    int MambaS6Plugin::getNbOutputs() const noexcept
    {
        return 1;
    }

    // 輸出維度與輸入 u 相同 [Batch, Length, D_model]
    DimsExprs MambaS6Plugin::getOutputDimensions(int outputIndex, const DimsExprs *inputs, int nbInputs, IExprBuilder &exprBuilder) noexcept
    {
        return inputs[0];
    }

    // 支援格式：我們只支援 FP16 (Half) 以達到 Jetson Orin Nano 最佳效能
    bool MambaS6Plugin::supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs, int nbOutputs) noexcept
    {
        // Inputs: 0:u, 1:delta, 2:A, 3:B, 4:C, 5:D
        // Output: 0:y
        return inOut[pos].type == DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
    }

    void MambaS6Plugin::configurePlugin(const DynamicPluginTensorDesc *in, int nbInputs, const DynamicPluginTensorDesc *out, int nbOutputs) noexcept
    {
        // 可以在這裡檢查維度是否正確
    }

    size_t MambaS6Plugin::getWorkspaceSize(const PluginTensorDesc *inputs, int nbInputs, const PluginTensorDesc *outputs, int nbOutputs) const noexcept
    {
        return 0; // 我們的 Kernel 不需要額外的 Workspace，因為用了 Shared Memory
    }

    // 核心執行邏輯
    int MambaS6Plugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
    {

        // 解析維度
        // inputDesc[0] is u: [Batch, Length, D_model]
        const int batch_size = inputDesc[0].dims.d[0];
        const int seq_len = inputDesc[0].dims.d[1];
        const int d_model = inputDesc[0].dims.d[2];

        // 呼叫 CUDA Kernel
        mambaS6KernelLauncher(
            static_cast<const half *>(inputs[0]), // u
            static_cast<const half *>(inputs[1]), // delta
            static_cast<const half *>(inputs[2]), // A
            static_cast<const half *>(inputs[3]), // B
            static_cast<const half *>(inputs[4]), // C
            static_cast<const half *>(inputs[5]), // D
            static_cast<half *>(outputs[0]),      // y
            batch_size,
            seq_len,
            d_model,
            mDState,
            stream);

        return 0;
    }

    // --- Boilerplate Code (序列化等) ---

    DataType MambaS6Plugin::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const noexcept
    {
        return DataType::kHALF;
    }

    const char *MambaS6Plugin::getPluginType() const noexcept { return "MambaS6Plugin"; }
    const char *MambaS6Plugin::getPluginVersion() const noexcept { return "1"; }
    void MambaS6Plugin::destroy() noexcept { delete this; }
    IPluginV2DynamicExt *MambaS6Plugin::clone() const noexcept
    {
        auto *plugin = new MambaS6Plugin(mName, mDState);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    void MambaS6Plugin::setPluginNamespace(const char *pluginNamespace) noexcept { mNamespace = pluginNamespace; }
    const char *MambaS6Plugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

    void MambaS6Plugin::serialize(void *buffer) const noexcept
    {
        char *d = static_cast<char *>(buffer);
        char *a = d;
        std::memcpy(d, &mDState, sizeof(mDState));
        d += sizeof(mDState);
        assert(d == a + getSerializationSize());
    }
    size_t MambaS6Plugin::getSerializationSize() const noexcept
    {
        return sizeof(mDState);
    }

    // --- Creator Implementation ---

    MambaS6PluginCreator::MambaS6PluginCreator()
    {
        mPluginAttributes.clear();
        mPluginAttributes.emplace_back(PluginField("d_state", nullptr, PluginFieldType::kINT32, 1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char *MambaS6PluginCreator::getPluginName() const noexcept { return "MambaS6Plugin"; }
    const char *MambaS6PluginCreator::getPluginVersion() const noexcept { return "1"; }
    const PluginFieldCollection *MambaS6PluginCreator::getFieldNames() noexcept { return &mFC; }

    IPluginV2 *MambaS6PluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
    {
        int d_state = 16; // Default
        const PluginField *fields = fc->fields;
        for (int i = 0; i < fc->nbFields; ++i)
        {
            const char *attrName = fields[i].name;
            if (!strcmp(attrName, "d_state"))
            {
                d_state = *(static_cast<const int *>(fields[i].data));
            }
        }
        return new MambaS6Plugin(name, d_state);
    }

    IPluginV2 *MambaS6PluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
    {
        return new MambaS6Plugin(serialData, serialLength);
    }

    void MambaS6PluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept { mNamespace = pluginNamespace; }
    const char *MambaS6PluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

    PluginFieldCollection MambaS6PluginCreator::mFC{};
    std::vector<PluginField> MambaS6PluginCreator::mPluginAttributes;

    REGISTER_TENSORRT_PLUGIN(MambaS6PluginCreator);

} // namespace mamba