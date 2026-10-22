# wotfkit

使用 JAX 计算三维弱散射传递函数，Python 导入名为 `wotf`。提供 FFT 互相关与 Ewald cap 两种算法，
默认双精度，可选单精度；参数显式传入，不依赖 `WOTFconfig`。

```bash
uv sync --locked
uv run --locked pytest -q
uv run --locked python examples/validate_born.py --dtype float64
uv run --locked python examples/validate_born.py --dtype float32
```

要求 Python ≥3.12，依赖由 `pyproject.toml` 和 `uv.lock` 管理，开发环境包括 pytest、Matplotlib。
仅安装运行依赖可用 `uv sync --locked --no-dev`。当前运行依赖固定为 `jax[cuda13]==0.12.0`，
安装会包含 CUDA 13 组件；GPU 的平台和驱动要求见 [JAX 官方安装说明](https://docs.jax.dev/en/latest/installation.html)。

## 使用

```python
import jax.numpy as jnp
from wotf import precision, k_axes, circular_pupil, compute_wotf, predict_contrast

shape = (33, 31, 35)                  # (nz, ny, nx)
spacing = (0.35, 0.20, 0.18)          # (dz, dy, dx), 样品空间 um

# 精度只选一次；改为 "float32" 即使用单精度。
with precision("float64"):
    kz, ky, kx = k_axes(shape, spacing)  # rad/um，未 shift
    # 照明 NA = 0.35，探测 NA = 0.6；照明只传强度 source。
    source = (
        ky[:, None]**2 + kx[None, :]**2 <= (2 * jnp.pi * 0.35 / 0.53)**2
    ).astype(ky.dtype)
    pupil = circular_pupil(ky, kx, wavelength=0.53, na=0.6)
    F_real = jnp.zeros(shape).at[16, 15, 17].set(-0.01)
    F_imag = jnp.zeros_like(F_real)

    optics = dict(wavelength=0.53, n_medium=1.33, source=source, pupil=pupil)
    H_real, H_imag = compute_wotf(shape, spacing, **optics)  # method="fft"
    H_ewald = compute_wotf(shape, spacing, **optics, method="ewald", axial="finite")
    c = predict_contrast(F_real, F_imag, H_real, H_imag)
    # absolute_intensity = I0 * (1 + c), I0 = sum(source * abs(pupil)**2)
```

`compute_wotf` 默认 `method="fft"`；选择 `method="ewald"` 后可用 `axial="finite"`（默认）
或 `axial="linear"`。`axial` 在 FFT 方法下不生效。仍可从 `wotf` 直接导入 `wotf_fft`、`wotf_ewald` 调用。

数值函数返回 `jax.Array`。无上下文时默认双精度：实数为 `float64`，传递函数为 `complex128`。
`with precision("float32"):` 中使用 `float32/complex64`，也会让块内普通 JAX 数组默认使用单精度。
精度优先级为**单次调用的 `dtype=` → 当前 `precision` 上下文 → 默认双精度**；
`dtype=None` 表示沿用上下文。上下文可嵌套，退出时恢复原精度和 JAX x64 配置，包括异常退出。
导入和调用不会永久改变 JAX 的 x64 配置。底层使用 [JAX 的局部 x64 上下文](https://docs.jax.dev/en/latest/_autosummary/jax.enable_x64.html)。
调用前已在单精度中创建的输入，其丢失的信息无法通过后续双精度转换恢复。
接口按课题组内部使用设计，假定输入满足下述形状、单位和取值约定；不逐项检查类型、有限性或 Hermitian 对称性。
WOTF 计算仅保留非零背景、传播模式和 Nyquist 检查。内部固定形状计算核使用 JIT。

## 定义与数组约定

- `wavelength` 是真空波长，所有长度单位一致；不隐式应用相机像元尺寸或放大倍数。
- `F = (2π/wavelength)² * (n_medium² − n²) = F_real + i F_imag`，单位为长度⁻²。
  正吸收对应负的 `F_imag`。
- 空间数组形状为 `(nz,ny,nx)`，坐标为 `(arange(n)−n//2)*d`。
  `H_real`、`H_imag` 是未 shift 的 FFT 频率排列，单位为长度²。
- 所有频率坐标使用角空间频率 `k`，单位 rad/长度；`k0=2π/wavelength`、`km=n_medium*k0`。
  傅里叶变换为 `F_hat(k)=∫F(r)exp(-ik·r)dr`，逆变换带 `1/(2π)³`。
  `k_axes` 返回 `2π*fftfreq(n,d)`，频率步长 `dk=2π/(n*d)`、Nyquist 为 `π/d`。
  预测使用中心空间数组与未 shift 频谱之间的显式转换。
- 两通道对应**散射势的实部和虚部**，不是某一个复传递函数的 `.real` 与 `.imag`。
- `source` 是 `(ny,nx)` 的非负非相干模式强度权重，`pupil` 是同形状的探测复振幅数组，表示孔径、振幅透过率和像差。
  按 `dky*dkx` 定义的连续光源密度采样值应先乘 `dky*dkx`；不传入场振幅。照明方向必须位于该频率网格。
- 照明与探测范围独立，不要求 NA 相同。`pupil == 0` 处的光源权重会被清零；确有正权重被丢弃时发出一次 `UserWarning`。
  清零后再计算背景和检查传播、采样条件；全部被丢弃时因背景为零而报错。非零复 pupil 不作阈值截断。
  孔径外照明没有检测到的未散射参考光，在线性阶不贡献对比度；其二次散射强度不由 WOTF 描述。
- 预测 `c=(I−I0)/I0`，`I0=sum(source*abs(pupil)**2)`；不减去样品平均值。
  体素 Fourier 积分因子在预测的变换与逆变换中约去，**不要再乘体素体积**。

FFT 与默认 finite Ewald 计算同一个有限轴向 lag 响应，浮点数组不保证逐位相等。
`wotf_ewald(..., axial="linear")`
另外提供原脚本的轴向线性落格思想，校正了显式单位和反向频率处理；它具有可测的落格误差。
两种方法都检查物理频率差是否满足 Nyquist，不通过取模隐藏欠采样。
绘图时可对 H 使用 `fftshift` 将零频移到中央；传入 `predict_contrast` 时仍使用未 shift 的 H。

FFT 的中间变量按 `analytical.tex` 组织：`illumination_arm` 对应光源加权、经过探测 pupil 的参考场，
`scattering_arm` 对应包含 `1/(2ik_z)` 的散射臂，其中 `k_z=sqrt(km²−kx²−ky²)`。
`A=corr(illumination_arm, scattering_arm)`、`B=corr(scattering_arm, illumination_arm)`，
轴向变换前的通道为 `h_real=A+B`、`h_imag=i(A−B)`，最后沿 z 做 FFT、乘 dz、除以 I₀。
两臂均去除共同载波 `exp(ikm z)` 以改善单精度稳定性；它在 A/B 中严格抵消。

Ewald 的单边系数为 `source*conj(P_source)*P_detector/(2*kz_detector*I0)`，
轴向核为 `D(Δkz,kz)=dz*Σ exp[i(Δkz−kz)*z_lag]`。无限范围时 `D→2πδ(kz−Δkz)`，
因此线性落格的总核权重为 `2π/dkz=nz*dz`，不能仅改为 `1/dkz`。

旧代码迁移：网格接口统一为 `k_axes`。旧 `frequency_axes` 的 cycles/长度坐标需乘 `2π`；
`wavevector_axes` 已经使用角空间频率，迁移时只需更名。自行构造 pupil 时截止为 `2π*NA/wavelength`。
空间采样、波长、散射势和 WOTF 的物理单位及预测公式不变。

计算代码按用途拆分：

- `src/wotf/wotf_fft.py`：FFT 互相关和 analytical 的 A/B 通道组合。
- `src/wotf/wotf_ewald.py`：Ewald cap、有限轴向核、线性落格及显式实部/虚部通道组合。
- `src/wotf/forward.py`：统一入口 `compute_wotf` 和对比度预测 `predict_contrast`。

日常调用继续使用 `from wotf import ...`；原混合模块不再保留。

## 测试

`tests/` 按目的拆为五个脚本，每个脚本保留一项测试：

| 脚本 | 检查内容 |
| --- | --- |
| [test_precision.py](tests/test_precision.py) | 默认双精度、单双精度切换、显式覆盖和 JAX x64 配置恢复；覆盖 FFT、finite、linear。 |
| [test_born.py](tests/test_born.py) | 弱 Gaussian 散射势的纯实、纯虚、混合情况；FFT 和 finite 与独立双精度标量 Born 场传播比较。 |
| [test_source_clipping.py](tests/test_source_clipping.py) | 超出 pupil 的照明自动丢弃、一次 warning、结果与手动清零一致，以及全部被丢弃后的零背景错误。 |
| [test_axial_sampling.py](tests/test_axial_sampling.py) | 从实际波矢差计算允许的 dz，确认上限以下可计算、上限以上三种路径均报错。 |
| [test_bead_forward.py](tests/test_bead_forward.py) | 小球 WOTF 前向预测非零且有限、通道可叠加、焦平面相位响应的奇对称和吸收响应的偶对称及负均值。 |

可单独运行，例如 `uv run --locked pytest tests/test_born.py -q`；运行全部测试使用 `uv run --locked pytest -q`。

Born 测试比较对比度而非总强度，相对 L2 阈值为双精度 `1e-10`、单精度 `1e-4`；
参考模型自行生成网格、逐层积分和传播，不调用 WOTF 内部函数。linear 的落格误差在独立示例中单独报告。
轴向 Nyquist 条件为 `max(abs(kz_detector-kz_source)) < pi/dz`：增加 nz 不会修复 dz 过大的欠采样，
nz 影响轴向频率步长与 lag 范围。

## 半圆与半环照明示例

```bash
uv run --locked python examples/half_patterns.py
uv run --locked python examples/half_patterns.py --method ewald --axial finite
```

探测使用 NA=0.75 的完整圆孔，照明分别是 NA≤0.75 的半圆盘和 0.5≤NA≤0.75 的半环，
都取 `ky >= 0` 的上半平面，包括直径边界。真空波长为 0.53 μm，介质折射率为 1.33，
网格 `(129,129,129)`、样品空间采样 `(0.35,0.15,0.15)` μm；横向可容纳照明与探测的最大波矢差。
默认同时计算 FFT 和线性落格 Ewald，使用双精度；可通过 `--dtype float32` 切换精度，`--output PATH` 指定输出目录。
`--method ewald --axial finite` 将第一套结果改为 finite Ewald，第二套仍为 linear Ewald。

`artifacts/half-patterns/` 下生成 `patterns.png`、`transfer_functions.png` 和 `transfer_functions_linear.png`。
后两幅图分别显示第一套方法与 linear Ewald 的两组 WOTF，使用相同布局和坐标范围；
均展示 `kz=0`、`kx=0`、`ky=0` 三个正交切片，同一通道在两种图案、两种方法之间共用色标，单位 μm²。
各切片按实际频率坐标等比例显示，`kz` 与 `kx`、`ky` 每 rad/μm 占相同长度；轴向切片自然较扁。
通道使用分组标题，光源使用列标签，切片平面标在行侧，避免重复标题。
色标采用对称对数映射、零点附近线性，以同时显示强 DC 与较弱的 cap 结构，保留数值和正负号。
本例采用实 pupil 和奇数轴向网格，`H_real` 为纯虚、`H_imag` 为纯实（浮点误差除外），
因此显示带符号的 `Im(H_real)`、`Re(H_imag)`，并打印另两个分量的残差；两者仍分别作用于散射势实部、虚部。
图中频谱经过 `fftshift`，只裁剪显示范围，计算结果保留完整的未 shift 频谱。
两个光源的非零模式权重均为 1，各自使用真实背景 `I0=sum(source*abs(pupil)**2)` 归一化。
linear 与 finite/FFT 对应不同的轴向离散，图中差异包含线性落格误差。

## 小球 forward 示例

```bash
uv run --locked python examples/forward_bead.py
uv run --locked python examples/forward_bead.py --method ewald --axial linear
```

半径 0.75 μm 的均匀小球位于样品中央，`n_bead=1.331+2e-5j`、背景 `n_medium=1.33`，真空波长为 0.53 μm。
散射势由 `F=(2π/λ)²*(n_medium²-n_bead²)` 得到；照明是 NA=0.75 的上半圆盘，探测是同 NA 的完整圆孔。
网格 `(65,65,65)`、采样 `(0.25,0.15,0.15)` μm；横向周期边界，轴向补零并只展示中央有效区域。
默认 FFT、双精度，可选 `--dtype float32`、`--method ewald --axial finite|linear`。

输出 `artifacts/bead/bead_forward.png` 与 `bead_stack.npz`。图中两行分别为 `F_real` 和预测对比度，
三列为正交切片，所有坐标按实际 μm 等比例显示，标题只标在首行。
NPZ 包含 `F_real`、`F_imag`、`contrast`、`intensity`、`background` 和 `(dz,dy,dx)`；
其中 `c=(I-I0)/I0`、`I=I0*(1+c)`。这项测试展示 WOTF 正向流程，不要求与精确球体散射或 Mie 理论一致。

## 独立物理验证与边界

示例的 Born 参考逐层计算散射场，在样品外出射面累加，再传播到九个焦平面；不复用 WOTF 代码。
同时比较线性强度与含 `|us|²` 的一阶 Born 强度。`--output PATH` 可指定输出目录，默认输出
`artifacts/born-float64/` 或 `artifacts/born-float32/` 中的 `metrics.json`、`comparison.png`、`profiles.png`。
参考统一使用双精度，便于独立衡量单精度预测误差；JSON 另外报告 FFT 与 finite Ewald 两个完整传递通道的差异。

横向采用周期样品；3D 频域预测本身是循环卷积。有限轴向样品需补零，使观察位置与样品位置之差
全部落在计算的 lag 范围内，再裁剪有效区域。示例物体仅占中央五层，比较中央九个焦平面。
焦点扫描是透射场数值重聚焦，不是样品内部完整场。

完整公式核查、原 solver 的符号解释、轴向离散差异和未解决问题见
[理论核查](docs/theory_audit.md)，实测误差和测试结果见 [验证报告](docs/validation_report.md)。
该模型适用于标量、传播波、单次弱散射；线性 WOTF 还忽略二次散射场强度。
