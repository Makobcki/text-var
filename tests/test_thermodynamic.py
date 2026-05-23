import torch
from src.var.generator import thermodynamic_sampling_with_stats


def test_mutation():
    print("[TEST] Тестирование Thermodynamic Sampling...")

    # Создаем фейковый словарь на 5 токенов
    vocab_size = 5
    batch_size = 2

    # Ситуация 1: Абсолютная уверенность модели (огромный разрыв между топ-1 и топ-2)
    confident_logits = torch.tensor([[100.0, 2.0, 1.0, 0.0, -5.0], [50.0, 1.0, 0.0, 0.0, 0.0]])

    # Ситуация 2: Модель сильно сомневается (разрыв минимален)
    uncertain_logits = torch.tensor([[10.0, 9.95, 9.8, 9.5, 9.0], [5.0, 4.99, 4.8, 4.0, 3.0]])

    # Тестируем уверенные логиты
    # Ожидаем, что температура упадет/приблизится к t_min, и сэмплер выберет строго токен 0
    sampled_confident, _, _ = thermodynamic_sampling_with_stats(
        confident_logits, alpha=1.0, t_min=0.1, t_max=2.0
    )
    print(f" Сэмплы при высокой уверенности (ожидаем [0, 0]): {sampled_confident.tolist()}")

    # Тестируем неуверенные логиты
    # Из-за маленького дельта температура поднимется до t_max (2.0), распределение размажется
    # Запустим 1000 раз, чтобы проверить вариативность (исследование латентного пространства)
    samples = []
    for _ in range(500):
        s = thermodynamic_sampling_with_stats(uncertain_logits, alpha=1.0, t_min=0.1, t_max=2.0)
        samples.extend(s[1].flatten().tolist())

    unique_tokens = set(samples)
    print(f" Уникальные токены при сомнении (ожидаем разнообразие > 1 токена): {unique_tokens}")
    assert len(unique_tokens) > 1, (
        "Ошибка: Динамическая температура не размывает распределение при сомнении!"
    )
    print("[SUCCESS] Мутация успешно протестирована и работает корректно!")


if __name__ == "__main__":
    test_mutation()
