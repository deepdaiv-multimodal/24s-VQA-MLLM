import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from .utils.load_data import CommonData, DataSet
from .model.beit3 import BEiT3ForVisualQuestionAnswering
import deepspeed
import logging
import json
logging.getLogger().setLevel(logging.ERROR)

class Runner:
    def __init__(self, __C, evaluator):
        self.__C = __C
        self.evaluator = evaluator

    def train(self, train_set, valid_set):
        net = BEiT3ForVisualQuestionAnswering(self.__C, num_classes=train_set.ans_size)

        # Load pretrained weights
        if os.path.exists(self.__C.PRETRAINED_MODEL_PATH):
            checkpoint = torch.load(self.__C.PRETRAINED_MODEL_PATH, map_location="cuda")
            net.load_state_dict(checkpoint, strict=False)
            print(f"Loaded pretrained weights from {self.__C.PRETRAINED_MODEL_PATH}")

        # 기본 AdamW 옵티마이저 설정
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, net.parameters()), 
            lr=self.__C.LR_BASE, 
            betas=self.__C.OPT_BETAS, 
            weight_decay=self.__C.WEIGHT_DECAY
        )

        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=net,
            optimizer=optimizer,
            config=self.__C.deepspeed_config
        )

        dataloader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8)
        criterion = torch.nn.BCEWithLogitsLoss()  # BCEWithLogitsLoss를 사용합니다.

        for epoch in range(self.__C.EPOCHS):
            model_engine.train()
            correct = 0
            total = 0
            
            for step, input_tuple in enumerate(dataloader):
                img, ques_ids, target = input_tuple

                img = img.cuda()
                ques_ids = ques_ids.cuda()
                target = target.cuda().float()  # target 텐서를 float 타입으로 변환

                optimizer.zero_grad()
                pred = model_engine(img, question=ques_ids)

                loss = criterion(pred, target)
                model_engine.backward(loss)
                model_engine.step()

                # 정확도 계산
                predicted = (torch.sigmoid(pred) > 0.5).float()  # BCE에서는 sigmoid를 사용하여 예측 값을 얻음
                correct += (predicted == target).sum().item()
                total += target.numel()

                if step % 2000 == 0:
                    print(f'Epoch {epoch}, Step {step}, Loss {loss.item()}')

            accuracy = 100 * correct / total
            print(f'Epoch {epoch}, Accuracy: {accuracy:.2f}%')

            # 모델 체크포인트 저장
            if not os.path.exists(self.__C.CKPTS_DIR):
                os.makedirs(self.__C.CKPTS_DIR)
            checkpoint_path = os.path.join(self.__C.CKPTS_DIR, f'epoch{epoch + 1}.pth')
            model_engine.save_checkpoint(checkpoint_path)
            print(f'Model checkpoint saved to {checkpoint_path}')

            # 평가 수행
            self.evaluate(valid_set, model_engine)

    @torch.no_grad()
    def evaluate(self, valid_set, model_engine):
        model_engine.eval()
        dataloader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=8)

        results = []
        correct = 0
        total = 0

        for step, input_tuple in enumerate(dataloader):
            img, ques_ids, target = input_tuple

            img = img.cuda()
            ques_ids = ques_ids.cuda()
            target = target.cuda().float()

            pred = model_engine(img, question=ques_ids)
            predicted = (torch.sigmoid(pred) > 0.5).float()
            correct += (predicted == target).sum().item()
            total += target.numel()

            # question_id와 단일 answer 저장
            question_id = int(ques_ids[0].item())
            answer_idx = torch.argmax(predicted).item()
            answer_word = self.__C.ANSWER_LIST[answer_idx]

            results.append({"question_id": question_id, "answer": answer_word})

        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')

        # 평가 결과를 파일에 저장
        results_path = self.__C.RESULT_PATH
        if not os.path.exists(os.path.dirname(results_path)):
            os.makedirs(os.path.dirname(results_path))
        with open(results_path, 'w') as f:
            json.dump(results, f)
        print(f'Evaluation results saved to {results_path}')

    def run(self):
        common_data = CommonData(self.__C)
        train_set = DataSet(self.__C, common_data, self.__C.TRAIN_SPLITS)
        valid_set = DataSet(self.__C, common_data, self.__C.EVAL_SPLITS)
        self.train(train_set, valid_set)
