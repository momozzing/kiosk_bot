# kiosk_bot

KoGPT2를 이용한 간단한 kiosk chatbot 입니다. 

데이터는 AiHub의 [한국어대화 데이터](https://aihub.or.kr/aidata/85)를 사용했습니다. 

데이터는 학습만 진행하였고 공개는 하지 않습니다. 
## Architecture 
Hugging face의 예제들을 보며 구현하였습니다. 

<img width="549" alt="gpt" src="https://user-images.githubusercontent.com/60643542/142431681-85db3d74-172d-45f0-9433-de43a8aeae17.png">

 ```input``` 은 ```User + <BOS> + System + <EOS>```로 구현하였다. 

```output``` 은 ```P(system|user)```를 최대화 시키기위해 모델을 학습하였다. 

## Install 
```
git clone https://github.com/momozzing/kiosk_bot.git
git lfs pull     ## lfs 가 없으면 설치하셔야 합니다. 
pip install requirements.txt
```

## How to train?
기존 학습된 데이터로 대화를 하고 싶으시면 ```install``` 후 ```How to chat``` 으로 넘어가셔도 됩니다. 
```
cd kiosk_bot/data  ## 안에 원하시는 데이터를 넣으시면 됩니다. 

kiosk_bot/GPT-2_fine_tune.py  ## 안에 데이터 loader들의 파일이름을 변경해주시면 됩니다. 

deepspeed --num_gpus=1 GPT-2_fine_tune.py ## 변경후 이것으로 실행하시면 됩니다. 

그후 생성된 skt-kogpt2-base-v2.pt 를 사용하시면 됩니다. 
```

## How to chat?
```
python interactive.py   ##기존 데이터로 채팅하기

python new_data_interactive.py  ##새로운 데이터로 파인튜닝한것 pt파일 불러와서 채팅하기.

--기존 데이터 채팅--

User: 아메리카노 주세요
System: 따뜻한 거 드릴까요?

User: 아이스로 주세요
System: 사이즈는 톨 괜찮으세요?

User: 그란데 사이즈로 주세요
System: 죄송한데 저희 가게에는 레귤러랑 라지 사이즈 밖에 없어요

User:  그럼 라지사이즈로 주세요
System: 네 고객님 테이크아웃 하시나요?

User: 먹고갈게요
System: 네 그럼 머그잔으로 준비해드리겠습니다

User: 감사합니다
System: 좋은 하루 보내세요
```

## Reference
[HuggingFace](https://huggingface.co/transformers/index.html)

[KoGPT2](https://github.com/SKT-AI/KoGPT2)

[AIHUB](https://aihub.or.kr/)
