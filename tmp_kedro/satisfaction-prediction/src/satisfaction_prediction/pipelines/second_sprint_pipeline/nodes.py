"""
This is a boilerplate pipeline 'second_sprint_pipeline'
generated using Kedro 1.0.0
"""

import wandb
from sklearn.metrics import f1_score


def train_baseline(X_train, y_train, params):
    """
    Funkcja trenuje podstawowy model (tzw. baseline).
    Jako argumenty przyjmuje dane treningowe X_train, y_train oraz parametry modelu.
    """

    # Uruchamiamy nowy eksperyment w W&B
    # Wartość "project" to nazwa projektu naszego zespołu w serwisie W&B
    # Wartość "config=params" pozwala zapisać w logach jakie hiperparametry miały wpływ na wynik
    wandb.init(project="asi-ml", job_type="train", reinit=True, config=params)

    # Tutaj odbywa się właściwy trening modelu.
    # W tym miejscu zespół wstawia kod, który tworzy i trenuje model np.:
    # model = LogisticRegression(**params)

    # model.fit(X_train, y_train)

    model = ...  # w tym miejscu powstaje wytrenowany model

    # W razie potrzeby można tu też zalogować wynik treningu na zbiorze uczącym:
    # wandb.log({"train_score": model.score(X_train, y_train)})

    # Nie zamykamy jeszcze sesji wandb.finish(),
    # bo chcemy, żeby metryki z ewaluacji modelu zapisały się w tym samym runie.
    # Zakończenie logowania nastąpi w funkcji evaluate().
    return model


def evaluate(model, X_test, y_test):
    """
    Funkcja ocenia model na zbiorze testowym.
    Oblicza metrykę F1 i zapisuje ją do W&B.
    Zwraca też metrykę jako słownik (dict), żeby Kedro mógł ją zapisać do pliku z wynikami.
    """

    # Przewidujemy wyniki dla danych testowych
    y_pred = model.predict(X_test)

    # Obliczamy miarę F1 (czyli jak dobrze model przewiduje klasy)
    f1 = f1_score(y_test, y_pred)

    # Zapisujemy metrykę F1 do aktualnie otwartego runa w W&B
    wandb.log({"F1_score": f1})

    # Kończymy runa (zamyka logowanie w W&B)
    wandb.finish()

    # Zwracamy wynik jako słownik
    # Dzięki temu Kedro będzie mógł zapisać wynik w pliku metrics_baseline.json
    return {"F1_score": float(f1)}
