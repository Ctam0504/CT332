# starcraft2-agents

Project mẫu cho khóa học: huấn luyện RL agents trên StarCraft II (PySC2)

**Maps**: MoveToBeacon, CollectMineralShards

## Yêu cầu

- StarCraft II (phiên bản tương thích với pysc2)
- Python 3.8+
- X server nếu chạy headful, hoặc chạy headless (use `DISPLAY`/xvfb)

## Cách cài

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Nếu cần: cài SC2 theo hướng dẫn pysc2 README
```
## Cách train
- Vào đường dẫn \CT332\starcraft_rl_project
- Câu lệnh train: python -m tênthưmục.tênfile
- Ví dụ khi muốn train map beacon với mô hình ppo thì sử dụng câu lênh: python -m train.train_beacon_ppo
- Sau khi train xong thì model sẽ lưu ở models\tên mô hình\
- Còn reward thì lưu ở logs\
