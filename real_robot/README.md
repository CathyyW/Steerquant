# SteerQuant real-robot evaluation

FastWAMJoint W4A8 deployment for the pack and stack tasks.

Visit our [project website](https://cathyyw.github.io/Steerquant-Website/) to see successful real-world robot demonstrations!🎉

## 1. Install

**Requirements:** Linux, Python 3.10, PyTorch 2.7.1, CUDA Toolkit 12.8,
a C++ compiler, Git, and Git LFS.

```bash
cd SteerQuant-copy/real_robot

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

git clone https://github.com/yuantianyuan01/FastWAM.git third_party/FastWAM
git -C third_party/FastWAM checkout 45d8e1458921d83f8ad6cf9ce993d371208dabd0
python -m pip install -e third_party/FastWAM

git clone https://github.com/NVIDIA/cutlass.git third_party/cutlass
git -C third_party/cutlass checkout da5e086dab31d63815acafdac9a9c5893b1c69e2

python -m pip install --no-deps -e .
```

The CUDA extension compiles on first use.
Live deployment also requires ROS 1, camera topics, configured `can_left` and
`can_right` SocketCAN interfaces, and `pyAgxArm`.

## 2. Download a W4A8 checkpoint

| Task | W4A8 checkpoint |
| --- | --- |
| pack | [ModelScope](https://modelscope.cn/datasets/cathyyw/pack_w4a8) |
| stack | [ModelScope](https://modelscope.cn/datasets/cathyyw/stack_w4a8) |

Download the selected task checkpoint and verify its checksum:

```bash
git clone https://www.modelscope.cn/datasets/cathyyw/pack_w4a8.git /tmp/pack_w4a8
git clone https://www.modelscope.cn/datasets/cathyyw/stack_w4a8.git /tmp/stack_w4a8

cp /tmp/pack_w4a8/deployment.pt checkpoints/pack_w4a8/
cp /tmp/stack_w4a8/deployment.pt checkpoints/stack_w4a8/

(cd checkpoints/pack_w4a8 && sha256sum -c SHA256SUMS)
(cd checkpoints/stack_w4a8 && sha256sum -c SHA256SUMS)
(cd assets && sha256sum -c SHA256SUMS)
```

BF16 baselines: [pack (step 15k)](https://www.modelscope.cn/models/Mikuz12/joint_pack/files)
· [stack (step 15k)](https://www.modelscope.cn/models/Mikuz12/joint_stack/files).

## 3. Select a task and validate the deployment

Copy the corresponding configuration template:

```bash
cp configs/robot_pack_w4a8.example.yaml configs/robot_pack_w4a8.local.yaml
# or: cp configs/robot_stack_w4a8.example.yaml configs/robot_stack_w4a8.local.yaml
```

Run the offline startup check (no robot required):

```bash
mkdir -p outputs
steerquant-serve-real-policy \
  --config configs/robot_pack_w4a8.local.yaml \
  --report outputs/pack_startup.json \
  --offline-only
```

## 4. Start the policy server

Set an authentication key and start the policy server:

```bash
read -rsp 'FASTWAM_AUTHKEY: ' FASTWAM_AUTHKEY; echo
export FASTWAM_AUTHKEY

steerquant-serve-real-policy \
  --config configs/robot_pack_w4a8.local.yaml \
  --report outputs/pack_startup.json
```

Use the stack configuration and report name when evaluating stack.

## 5. Connect the robot controller

Keep the server running. In a second terminal, use the same Python environment
and authentication key, and source the ROS workspace. Start the camera topics
and SocketCAN interfaces first.

Copy the live configuration and check the ROS topics, CAN channels, initial
poses, and safety limits for your robot:

```bash
cp configs/robot_pack_w4a8_live.example.yaml \
  configs/robot_pack_w4a8_live.local.yaml
# or use robot_stack_w4a8_live.example.yaml
```

Run the client in shadow mode first:

```bash
source .venv/bin/activate
source /opt/ros/noetic/setup.bash
# source /path/to/camera_ws/devel/setup.bash

read -rsp 'FASTWAM_AUTHKEY: ' FASTWAM_AUTHKEY; echo
export FASTWAM_AUTHKEY

steerquant-run-real-robot \
  --config configs/robot_pack_w4a8_live.local.yaml \
  --max-steps 128
```

Shadow mode reads observations without moving the robot.
After checking shadow mode and the physical workspace, enable execution:

```bash
steerquant-run-real-robot \
  --config configs/robot_pack_w4a8_live.local.yaml \
  --max-steps 100000 \
  --execute-actions \
  --move-to-initial
```

Type `EXECUTE` to start. Space stops the policy loop, not an emergency stop.
Use `--move-to-initial` unless the robot is already at a suitable starting pose.

Camera input: `uint8 [480, 640, 3]`, with no RGB/BGR conversion.
Robot state: 14 values. Actions: `[32, 14]`, consumed at 40 Hz.

## Fixed evaluation settings

| Setting | Value |
|---|---|
| Quantization | W4A8 (4-bit packed weights, 8-bit activations) |
| Denoising steps | 10 |
| Action horizon | 32 |
| Action dimension | 14 |
| Video frames | 9 |
| Other operators | BF16 |
| W4A8 kernel | CUTLASS tile 64 |
| Execution | CUDA Graph enabled |

