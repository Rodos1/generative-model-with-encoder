# generative-model-with-encoder
Это работа по применению D2NN в качестве оптической генеративной нейросети

В отличие от предыдущей реализации без encoder, здесь был реализован Conditional Phase Encoder, который преобразует нормальный шумовой вектор в фазовую карту

---------

В качестве оптимизатора использовался AdamW, функция ошибок: 80% L1Loss, 20% MSELoss

---------

Гиперпараметры:

latent_dim = 100

n = 64

batch_size = 1000

epochs_num = 15

lr = 0.001

num_classes = 10 (цифра от 0 до 9)

---------

model = configure_dnn(n=n, pixels=n, length=0.001, wavelength=500E-9, masks_amount=3, distance=0.08628599497985633, detectors_norm='none')

encoder = ConditionalPhaseEncoder(latent_dim, n, num_classes)

---------

Так же были произведены следующие нормировки:

Генерируемый шум (среднее - 0, std - 1) был ограничен в пределах от -2 до 2

После прохождения D2NN есть minmax нормировка в диапазон от 0 до 1

--------

Каждое изображение из оригинального MNIST было преобразован в тензор, а в процессе обучения метка (label) каждого изображения MNIST была преобразован в one-hot вектор 
([3] --> [0,0,0,1,0,0,0,0,0,0])

![15](https://github.com/user-attachments/assets/2f368d61-a8ce-449e-8f60-a51b0aaa848f)
![15, only 5](https://github.com/user-attachments/assets/83d92c5b-2e21-4ee6-9b83-70ec2581917a)

