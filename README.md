# Perfect Clear Solver – level 602

Desktopowa aplikacja na Windows, która podpowiada najlepszy ruch w poziomie wymagającym idealnego wyczyszczenia planszy: **35 klocków, 14 linii, pusta plansza na końcu**. Klikasz aktualny klocek i next, a program pokazuje, gdzie go położyć i jaka jest szansa na wygraną. Liczy wyłącznie na CPU.

> Projekt nieoficjalny, niezwiązany z The Tetris Company ani wydawcami gry. Tetris® jest zastrzeżonym znakiem towarowym jego właściciela. Repozytorium nie zawiera grafiki, muzyki ani kodu gry.

## Wyniki

Milion symulowanych partii na losowych sekwencjach 7-bag:

| Strategia | Przejście levelu |
| --- | ---: |
| Solver | **16,01% ± 0,07** |
| Zachłanna (niski, równy stos) | 0,057% ± 0,005 |
| Losowa | 0,000% |

Wartość policzona dokładnie z tabel solvera to 16,02%.

## Uruchomienie

Wymagany Python 3.12 i około 4 GB wolnego miejsca.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
python tetris_pc\app.py
```

Pierwsze uruchomienie liczy tabele przez około 8 minut na 16 wątkach i zapisuje je w `tetris_pc/cache`. Kolejne starty trwają kilka sekund.

## Obsługa

| Klawisz | Działanie |
| --- | --- |
| `I O T S Z J L` | aktualny klocek |
| `Shift` + litera | next |
| `Enter` | położyłem klocek |
| `Backspace` | cofnij |

Jeśli obrót w grze działa w lewo, ustaw `"rotation_tap": "left"` w `tetris_pc/config.json`. Plik powstaje przy pierwszym uruchomieniu.

## Jak to działa

- **Silnik:** plansza jako bitboard. Generator ruchów obraca klocek tylko nad stosem, a potem przesuwa go w bok i w dół, więc obsługuje wsunięcia pod nawis.
- **Solver:** expectimax po całych 35 klockach ze śledzeniem worka 7-bag i jednym widocznym nextem. Około 25 mln stanów, wartości zapisane na dysku, podpowiedź w mniej niż 1 ms.
- **Model gry:** dopóki zostało więcej niż 4 linie, stos ma najwyżej 4 wiersze i nie ma dziur. Nawisy i wsunięcia są dozwolone w ostatnich 5 klockach. Podane 16% to więc dokładna szansa w tym modelu i dolne ograniczenie szansy przy dowolnej grze.
- **Założenie:** pierwszy klocek levelu otwiera nowy worek. Inny stan worka można ustawić w aplikacji, a program policzy dla niego osobną tabelę.

## Testy i symulacja

```powershell
.venv\Scripts\python -m pip install pytest
.venv\Scripts\python -m pytest tetris_pc\tests
python tetris_pc\simulate.py --games 1000000 --policy all
```

## Budowa `.exe`

```powershell
.venv\Scripts\python -m pip install pyinstaller
.venv\Scripts\pyinstaller --noconfirm --windowed --onedir --name TetrisPCSolver --paths tetris_pc --collect-all numba --collect-all llvmlite --hidden-import psutil tetris_pc\app.py
```
