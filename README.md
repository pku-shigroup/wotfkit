# wotfkit

使用 JAX 计算三维弱散射传递函数，Python 导入名为 `wotf`。提供 FFT 互相关与 Ewald cap 两种算法，
默认双精度，可选单精度；参数显式传入，不依赖 `WOTFconfig`。

```bash
uv sync --locked
uv run --locked pytest -q
uv run --locked python examples/validate_born.py --dtype float64
uv run --locked python examples/validate_born.py --dtype float32
```

依赖由 `pyproject.toml` 和 `uv.lock` 管理，开发环境包括 pytest、Matplotlib。
仅安装运行依赖可用 `uv sync --locked --no-dev`。默认安装支持 CPU；GPU 环境需按
[JAX 官方安装说明](https://docs.jax.dev/en/latest/installation.html)选择对应加速依赖。

## 使用

```python
import jax
import jax.numpy as jnp
from wotf import frequency_axes, circular_pupil, wotf_fft, wotf_ewald, predict_contrast

shape = (33, 31, 35)                  # (nz, ny, nx)
spacing = (0.35, 0.20, 0.18)          # (dz, dy, dx), 样品空间 um
fz, fy, fx = frequency_axes(shape, spacing)  # cycles/um，未 shift
pupil = circular_pupil(fy, fx, wavelength=0.53, na=0.6)

# 应用侧创建双精度输入也应启用 x64；库的每次调用会自行管理局部上下文。
with jax.enable_x64(True):
    source = jnp.zeros(shape[1:], dtype=jnp.float64).at[0, 0].set(1)
    F_real = jnp.zeros(shape, dtype=jnp.float64).at[16, 15, 17].set(-0.01)
    F_imag = jnp.zeros_like(F_real)

optics = dict(wavelength=0.53, n_medium=1.33, source=source, pupil=pupil)
H_real, H_imag = wotf_fft(shape, spacing, **optics)
H_ewald = wotf_ewald(shape, spacing, **optics)  # axial="finite"，准确有限采样
c = predict_contrast(F_real, F_imag, H_real, H_imag)
# absolute_intensity = I0 * (1 + c), I0 = sum(source * abs(pupil)**2)

# 单精度：网格、pupil、WOTF 和预测接口均支持 dtype="float32"。
H_single = wotf_ewald(shape, spacing, **optics, dtype="float32")
c_single = predict_contrast(F_real, F_imag, *H_single, dtype="float32")
```

所有公开函数返回 `jax.Array`。默认实数为 `float64`，传递函数为 `complex128`；
选择 `dtype="float32"` 后分别为 `float32/complex64`。导入和调用不会永久改变 JAX 的 x64 配置。
调用前已在单精度中创建的输入，其丢失的信息无法通过后续双精度转换恢复。
公开入口进行具体数值校验，内部固定形状计算核使用 JIT；入口不承诺可直接被外层 `jax.jit` 追踪。

## 定义与数组约定

- `wavelength` 是真空波长，所有长度单位一致；不隐式应用相机像元尺寸或放大倍数。
- `F = (2π/wavelength)² * (n_medium² − n²) = F_real + i F_imag`，单位为长度⁻²。
  正吸收对应负的 `F_imag`。
- 空间数组形状为 `(nz,ny,nx)`，坐标为 `(arange(n)−n//2)*d`。
  `H_real`、`H_imag` 是未 shift 的 FFT 频率排列，单位为长度²。
- 傅里叶核为 `exp(-2πif·r)`。预测使用中心空间数组与未 shift 频谱之间的显式转换。
- 两通道对应**散射势的实部和虚部**，不是某一个复传递函数的 `.real` 与 `.imag`。
- `source` 是 `(ny,nx)` 的非负非相干模式强度权重，`pupil` 是同形状的复振幅 pupil。
  连续光源密度采样值应先乘 `dfy*dfx`；不传入场振幅。照明方向必须位于该频率网格。
- 预测 `c=(I−I0)/I0`，`I0=sum(source*abs(pupil)**2)`；不减去样品平均值。
  体素 Fourier 积分因子在预测的变换与逆变换中约去，**不要再乘体素体积**。

FFT 与默认 Ewald 计算同一个有限轴向 lag 响应。`wotf_ewald(..., axial="linear")`
另外提供原脚本的轴向线性落格思想，校正了显式单位和反向频率处理；它具有可测的落格误差。
两种方法都检查物理频率差是否满足 Nyquist，不通过取模隐藏欠采样。

FFT 的中间变量按 `analytical.tex` 组织：`illumination_arm` 对应光源加权的 pupil，
`scattering_arm` 对应包含 `1/(4πi w)` 的散射臂（文档角频率写法为 `1/(2ik_z)`）。
`A=corr(illumination_arm, scattering_arm)`、`B=corr(scattering_arm, illumination_arm)`，
轴向变换前的通道为 `h_real=A+B`、`h_imag=i(A−B)`，最后沿 z 做 FFT、乘 dz、除以 I₀。
两臂均去除共同载波 `exp(2πif_m z)` 以改善单精度稳定性；它在 A/B 中严格抵消。

## 独立物理验证与边界

示例的 Born 参考逐层计算散射场，在样品外出射面累加，再传播到九个焦平面；不复用 WOTF 代码。
同时比较线性强度与含 `|us|²` 的一阶 Born 强度。`--output PATH` 可指定输出目录，默认输出
`artifacts/born-float64/` 或 `artifacts/born-float32/` 中的 `metrics.json`、`comparison.png`、`profiles.png`。
参考统一使用双精度，便于独立衡量单精度预测误差。

横向采用周期样品；3D 频域预测本身是循环卷积。有限轴向样品需补零，使观察位置与样品位置之差
全部落在计算的 lag 范围内，再裁剪有效区域。示例物体仅占中央五层，比较中央九个焦平面。
焦点扫描是透射场数值重聚焦，不是样品内部完整场。

完整公式核查、原 solver 的符号解释、轴向离散差异和未解决问题见
[理论核查](docs/theory_audit.md)，实测误差和测试结果见 [验证报告](docs/validation_report.md)。
该模型适用于标量、传播波、单次弱散射；线性 WOTF 还忽略二次散射场强度。
