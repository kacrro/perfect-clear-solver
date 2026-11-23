"""Aplikacja desktopowa: podpowiedzi ruchów i symulacja dla levelu 602."""
from __future__ import annotations

from bootstrap import use_project_venv

if __name__ == "__main__":
    use_project_venv(__file__)

import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import psutil
from PySide6.QtCore import QObject, QPointF, QProcess, QRectF, QRunnable, Qt, QThread, QThreadPool, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from bag import FULL_BAG, remove_piece
from engine import NUM_ORI, PIECES, Move, clear_rows, piece_cells
from simulate import POLICY_NAMES, SimResult, simulate
from solver import APP_DIR, TOTAL_LINES, TOTAL_PIECES, BagMismatch, MoveEval, Solver, file_progress

CONFIG_PATH = APP_DIR / "config.json"
BOARD_ROWS = 6
EDITABLE_ROWS = 4
COLORS = {
    "I": "#35c9e8", "O": "#f2d13c", "T": "#a44fe0", "S": "#4fc94f",
    "Z": "#e8484a", "J": "#3f6fe8", "L": "#f0903a",
}
STYLE = """
QMainWindow, QWidget { background: #15171c; color: #e7e9ee; font-size: 14px; }
QTabWidget::pane { border: 1px solid #2b2f38; }
QTabBar::tab { background: #1e2128; padding: 8px 20px; border: 1px solid #2b2f38; }
QTabBar::tab:selected { background: #2b2f38; }
QPushButton { background: #2a2e37; border: 1px solid #3a3f4b; border-radius: 6px; padding: 8px 12px; }
QPushButton:hover { background: #363b47; }
QPushButton:checked { background: #4a5163; border-color: #9aa3b8; }
QPushButton:disabled { color: #6b7180; }
QComboBox, QSpinBox { background: #22252d; border: 1px solid #3a3f4b; border-radius: 4px; padding: 5px; }
QProgressBar { background: #22252d; border: 1px solid #3a3f4b; border-radius: 4px; text-align: center; }
QProgressBar::chunk { background: #3f8f6b; }
QStatusBar { background: #101216; }
"""


def load_config() -> dict:
    config = {"rotation_tap": "right"}
    try:
        config.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def percent(value: float) -> str:
    if value <= 0:
        return "0%"
    if value < 0.0001:
        return "<0,01%"
    return f"{value * 100:.2f}%".replace(".", ",")


def describe_move(move: Move, rotation_tap: str) -> str:
    """Instrukcja słowna, np. ``T: obróć 1× w prawo, kolumna 7, opuść``."""
    name = PIECES[move.piece]
    states = int(NUM_ORI[move.piece])
    if rotation_tap == "left" and states == 4:
        taps = (4 - move.orientation) % 4
    else:
        taps = move.orientation
    word = "w lewo" if rotation_tap == "left" else "w prawo"
    parts = [f"obróć {taps}× {word}" if taps else "bez obrotu"]
    parts.append(f"lewa krawędź w kolumnie {move.drop_x + 1}")
    for kind, amount in move.path:
        if kind == "drop":
            parts.append("opuść")
        else:
            parts.append(f"wsuń {abs(amount)} {'w lewo' if amount < 0 else 'w prawo'}")
    if not move.path:
        parts.append("opuść")
    return f"{name}: " + ", ".join(parts)


@dataclass(frozen=True)
class GameState:
    field: int = 0
    t: int = 0
    bag_before: int = FULL_BAG
    current: int | None = None
    nxt: int | None = None

    @property
    def pool(self) -> int:
        return self.bag_before or FULL_BAG

    @property
    def bag_after_current(self) -> int | None:
        return None if self.current is None else remove_piece(self.bag_before, self.current)

    @property
    def lines_cleared(self) -> int | None:
        total = 4 * self.t - bin(self.field).count("1")
        return total // 10 if total % 10 == 0 and total >= 0 else None


class BoardWidget(QWidget):
    cell_toggled = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.field = 0
        self.preview: list[tuple[int, int]] = []
        self.preview_color = "#ffffff"
        self.editing = False
        self.setMinimumSize(480, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def show_state(self, field: int, cells: list[tuple[int, int]], color: str) -> None:
        self.field, self.preview, self.preview_color = field, cells, color
        self.update()

    def _geometry(self) -> tuple[float, float, float]:
        size = min(self.width() / 10, self.height() / BOARD_ROWS)
        return size, (self.width() - size * 10) / 2, (self.height() - size * BOARD_ROWS) / 2

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        size, left, top = self._geometry()
        for row in range(BOARD_ROWS):
            for col in range(10):
                rect = QRectF(left + col * size, top + (BOARD_ROWS - 1 - row) * size, size, size).adjusted(1, 1, -1, -1)
                base = "#1c1f26" if row < EDITABLE_ROWS else "#17191f"
                painter.fillRect(rect, QColor(base))
                if row < EDITABLE_ROWS and (self.field >> (10 * row + col)) & 1:
                    painter.fillRect(rect, QColor("#7b8190"))
        color = QColor(self.preview_color)
        for col, row in self.preview:
            if 0 <= row < BOARD_ROWS:
                rect = QRectF(left + col * size, top + (BOARD_ROWS - 1 - row) * size, size, size).adjusted(2, 2, -2, -2)
                painter.fillRect(rect, color)
                painter.setPen(QPen(QColor("#ffffff"), 2))
                painter.drawRect(rect)
        painter.setPen(QPen(QColor("#596071"), 1, Qt.PenStyle.DashLine))
        y = top + (BOARD_ROWS - EDITABLE_ROWS) * size
        painter.drawLine(QPointF(left, y), QPointF(left + 10 * size, y))
        painter.setPen(QColor("#8b92a3"))
        painter.setFont(QFont(self.font().family(), 9))
        for col in range(10):
            painter.drawText(QRectF(left + col * size, top, size, 18), Qt.AlignmentFlag.AlignCenter, str(col + 1))

    def mousePressEvent(self, event) -> None:
        if not self.editing:
            return
        size, left, top = self._geometry()
        col = int((event.position().x() - left) // size)
        row = BOARD_ROWS - 1 - int((event.position().y() - top) // size)
        if 0 <= col < 10 and 0 <= row < EDITABLE_ROWS:
            self.cell_toggled.emit(col, row)


class PieceButton(QPushButton):
    def __init__(self, name: str, compact: bool = False):
        super().__init__(name)
        color = COLORS[name]
        self.setCheckable(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMinimumHeight(30 if compact else 46)
        self.setStyleSheet(
            f"QPushButton {{ background: {color}; color: #101216; font-weight: 700; border: 3px solid transparent; }}"
            "QPushButton:checked { border-color: #ffffff; }"
            "QPushButton:disabled { background: #2a2e37; color: #5c6270; }"
        )


class AnalyzeSignals(QObject):
    finished = Signal(int, object, float, str)


class AnalyzeTask(QRunnable):
    def __init__(self, request: int, solver: Solver, state: GameState):
        super().__init__()
        self.request, self.solver, self.state = request, solver, state
        self.signals = AnalyzeSignals()

    def run(self) -> None:
        started = time.perf_counter()
        error = ""
        results: list[MoveEval] = []
        try:
            results = self.solver.analyze(
                self.state.field, self.state.t, self.state.current, self.state.nxt, self.state.bag_after_current
            )
        except BagMismatch as exc:
            error = str(exc)
        self.signals.finished.emit(self.request, results, time.perf_counter() - started, error)


class SimulationThread(QThread):
    progressed = Signal(object)

    def __init__(self, solver: Solver, policy: str, games: int):
        super().__init__()
        self.solver, self.policy, self.games = solver, policy, games
        self.cancelled = False

    def run(self) -> None:
        def report(result: SimResult) -> bool:
            self.progressed.emit(replace(result))
            return not self.cancelled

        simulate(self.policy, self.games, on_progress=report, solver=self.solver)


class SimulationTab(QWidget):
    def __init__(self, solver: Solver):
        super().__init__()
        self.solver = solver
        self.thread_worker: SimulationThread | None = None
        self.results: dict[str, SimResult] = {}
        self.policy = QComboBox()
        for key, label in POLICY_NAMES.items():
            self.policy.addItem(label, key)
        self.games = QSpinBox()
        self.games.setRange(1000, 5_000_000)
        self.games.setSingleStep(10_000)
        self.games.setValue(100_000)
        self.games.setGroupSeparatorShown(True)
        self.start = QPushButton("Start")
        self.start.clicked.connect(self.toggle)
        self.bar = QProgressBar()
        self.summary = QLabel("Wybierz politykę i naciśnij Start.")
        self.summary.setWordWrap(True)
        self.summary.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.cpu = QLabel("CPU: –")
        controls = QHBoxLayout()
        for widget in (QLabel("Polityka"), self.policy, QLabel("Liczba gier"), self.games, self.start):
            controls.addWidget(widget)
        controls.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.bar)
        layout.addWidget(self.cpu)
        layout.addWidget(self.summary, 1)
        self.cpu_timer = QTimer(self)
        self.cpu_timer.timeout.connect(self.update_cpu)
        self.cpu_timer.start(1000)
        psutil.cpu_percent(percpu=True)

    def update_cpu(self) -> None:
        loads = psutil.cpu_percent(percpu=True)
        busy = sum(1 for load in loads if load > 50)
        self.cpu.setText(f"CPU: {sum(loads) / len(loads):.0f}%  (mocno pracujące wątki: {busy}/{len(loads)})")

    def toggle(self) -> None:
        if self.thread_worker is not None:
            self.thread_worker.cancelled = True
            self.start.setEnabled(False)
            return
        policy = self.policy.currentData()
        if policy == "optimal" and not self.solver.values_ready(0):
            self.summary.setText("Tabele solvera nie są jeszcze gotowe.")
            return
        self.bar.setRange(0, self.games.value())
        self.bar.setValue(0)
        self.start.setText("Stop")
        self.thread_worker = SimulationThread(self.solver, policy, self.games.value())
        self.thread_worker.progressed.connect(self.on_progress)
        self.thread_worker.finished.connect(self.on_finished)
        self.thread_worker.start()

    def on_progress(self, result: SimResult) -> None:
        self.bar.setValue(result.games)
        self.results[result.policy] = result
        self.render_results()

    def on_finished(self) -> None:
        self.thread_worker = None
        self.start.setText("Start")
        self.start.setEnabled(True)

    def render_results(self) -> None:
        rows = "".join(
            f"<tr><td>{POLICY_NAMES[r.policy]}</td><td align='right'>{r.games:,}</td>"
            f"<td align='right'><b>{r.win_rate * 100:.3f}%</b> ±{r.margin * 100:.3f}</td>"
            f"<td align='right'>{r.reached_rate * 100:.1f}%</td><td align='right'>{r.block_rate * 100:.1f}%</td>"
            f"<td align='right'>{r.decision_us:.1f} µs</td></tr>"
            for r in self.results.values()
        )
        self.summary.setText(
            "<table cellpadding='6'><tr><th align='left'>Polityka</th><th>Gry</th><th>% levelu</th>"
            f"<th>Doszło do 4 linii</th><th>% końcowego bloku</th><th>Czas decyzji</th></tr>{rows}</table>"
        )


class GameTab(QWidget):
    status = Signal(str)
    build_requested = Signal(int)

    def __init__(self, solver: Solver, config: dict):
        super().__init__()
        self.solver = solver
        self.config = config
        self.state = GameState()
        self.history: list[GameState] = []
        self.results: list[MoveEval] = []
        self.selected = 0
        self.request = 0
        self.animation: list[list[tuple[int, int]]] = []
        self.pool = QThreadPool.globalInstance()
        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(220)
        self.anim_timer.timeout.connect(self._animate)
        self._build_ui()
        self._bind_shortcuts()
        self.refresh()

    def _build_ui(self) -> None:
        self.board = BoardWidget()
        self.board.cell_toggled.connect(self.toggle_cell)
        self.instruction = QLabel("Wybierz klocek, który teraz spada.")
        self.instruction.setWordWrap(True)
        self.instruction.setMinimumHeight(54)
        self.instruction.setStyleSheet("font-size: 17px; font-weight: 600;")
        self.pending_offset = 0
        self.build_button = self._action("Policz tabele dla tego worka (ok. 1,5 min, 3,4 GB)", self.request_build)
        self.build_button.hide()
        left = QVBoxLayout()
        left.addWidget(self.board, 1)
        left.addWidget(self.instruction)
        left.addWidget(self.build_button)

        self.counters = QLabel()
        self.counters.setStyleSheet("font-size: 15px;")
        self.current_buttons = self._piece_row(self.set_current)
        self.next_buttons = self._piece_row(self.set_next)
        self.bag_buttons = self._piece_row(self.toggle_bag, compact=True)
        self.move_buttons: list[QPushButton] = []
        moves = QVBoxLayout()
        for i in range(3):
            button = QPushButton("–")
            button.setCheckable(True)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setStyleSheet("text-align: left;")
            button.clicked.connect(lambda _=False, index=i: self.select_move(index))
            self.move_buttons.append(button)
            moves.addWidget(button)
        self.placed = self._action("Położyłem ✓  (Enter)", self.place)
        self.placed.setMinimumHeight(46)
        self.placed.setStyleSheet("background: #2f7d5b; font-weight: 700;")
        self.undo_button = self._action("Cofnij", self.undo)
        self.reset_button = self._action("Reset", self.reset)
        self.path_button = self._action("Pokaż ścieżkę", self.play_path)
        self.edit_button = self._action("Edycja planszy", self.toggle_edit)
        self.edit_button.setCheckable(True)
        self.move_spin = QSpinBox()
        self.move_spin.setRange(1, TOTAL_PIECES)
        self.move_spin.setPrefix("Ruch ")
        self.move_spin.setEnabled(False)
        self.move_spin.valueChanged.connect(self.set_move_number)
        actions = QGridLayout()
        actions.addWidget(self.undo_button, 0, 0)
        actions.addWidget(self.reset_button, 0, 1)
        actions.addWidget(self.path_button, 0, 2)
        actions.addWidget(self.edit_button, 1, 0, 1, 2)
        actions.addWidget(self.move_spin, 1, 2)

        right = QVBoxLayout()
        right.addWidget(self.counters)
        right.addWidget(self._heading("Teraz spada  (litera)"))
        right.addLayout(self.current_buttons[0])
        right.addWidget(self._heading("Next  (Shift + litera)"))
        right.addLayout(self.next_buttons[0])
        right.addWidget(self._heading("Zostało w worku  (kliknij, aby poprawić)"))
        right.addLayout(self.bag_buttons[0])
        right.addWidget(self._heading("Najlepsze ruchy"))
        right.addLayout(moves)
        right.addWidget(self.placed)
        right.addLayout(actions)
        right.addStretch(1)
        panel = QFrame()
        panel.setLayout(right)
        panel.setFixedWidth(430)
        root = QHBoxLayout(self)
        root.addLayout(left, 1)
        root.addWidget(panel)

    @staticmethod
    def _heading(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("color: #9aa3b8; font-size: 12px; margin-top: 6px;")
        return label

    def _action(self, text: str, handler) -> QPushButton:
        button = QPushButton(text)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.clicked.connect(lambda _=False: handler())
        return button

    def _piece_row(self, handler, compact: bool = False):
        layout = QHBoxLayout()
        layout.setSpacing(4)
        buttons = []
        for index, name in enumerate(PIECES):
            button = PieceButton(name, compact)
            button.clicked.connect(lambda _=False, p=index: handler(p))
            layout.addWidget(button)
            buttons.append(button)
        return layout, buttons

    def _bind_shortcuts(self) -> None:
        for index, name in enumerate(PIECES):
            QShortcut(QKeySequence(name), self, activated=lambda p=index: self.set_current(p))
            QShortcut(QKeySequence(f"Shift+{name}"), self, activated=lambda p=index: self.set_next(p))
        for key in ("Return", "Enter"):
            QShortcut(QKeySequence(key), self, activated=self.place)
        QShortcut(QKeySequence("Backspace"), self, activated=self.undo)

    def _push(self, state: GameState) -> None:
        self.history.append(self.state)
        self.state = state
        self.refresh()

    def set_current(self, piece: int) -> None:
        if not (self.state.pool >> piece) & 1:
            self.status.emit(f"Klocka {PIECES[piece]} nie ma w worku. Popraw worek, jeśli to pomyłka.")
            self.refresh()
            return
        self._push(replace(self.state, current=piece, nxt=None))

    def set_next(self, piece: int) -> None:
        if self.state.current is None:
            self.status.emit("Najpierw wybierz klocek, który teraz spada.")
            self.refresh()
            return
        pool = self.state.bag_after_current or FULL_BAG
        if not (pool >> piece) & 1:
            self.status.emit(f"Next {PIECES[piece]} nie pasuje do worka. Popraw worek, jeśli to pomyłka.")
            self.refresh()
            return
        self._push(replace(self.state, nxt=piece))

    def toggle_bag(self, piece: int) -> None:
        state = self.state
        if state.current == piece:
            self.refresh()
            return
        mask = state.pool ^ (1 << piece)
        if state.current is not None:
            mask |= 1 << state.current
        nxt = state.nxt if state.nxt is None or ((remove_piece(mask, state.current) or FULL_BAG) >> state.nxt) & 1 else None
        self._push(replace(state, bag_before=mask, nxt=nxt))

    def toggle_cell(self, col: int, row: int) -> None:
        self._push(replace(self.state, field=self.state.field ^ (1 << (10 * row + col))))

    def toggle_edit(self) -> None:
        editing = self.edit_button.isChecked()
        self.board.editing = editing
        self.move_spin.setEnabled(editing)
        self.refresh()

    def set_move_number(self, value: int) -> None:
        if self.board.editing and value - 1 != self.state.t:
            self._push(replace(self.state, t=value - 1))

    def undo(self) -> None:
        if self.history:
            self.state = self.history.pop()
            self.refresh()

    def reset(self) -> None:
        self.history.clear()
        self.state = GameState()
        self.edit_button.setChecked(False)
        self.board.editing = False
        self.move_spin.setEnabled(False)
        self.refresh()

    def place(self) -> None:
        if not self.results or self.selected >= len(self.results):
            return
        move = self.results[self.selected].move
        state = self.state
        self._push(GameState(move.result_field, state.t + 1, state.bag_after_current, state.nxt, None))

    def select_move(self, index: int) -> None:
        if index < len(self.results):
            self.selected = index
        self.anim_timer.stop()
        self._render()

    def play_path(self) -> None:
        if not self.results:
            return
        move = self.results[self.selected].move
        lines = Solver.lines_remaining(self.state.field, self.state.t) or 4
        x, y = move.drop_x, min(lines, EDITABLE_ROWS)
        frames = [piece_cells(move.piece, move.orientation, x, y)]
        for kind, amount in move.path:
            for _ in range(abs(amount)):
                if kind == "drop":
                    y -= 1
                else:
                    x += 1 if amount > 0 else -1
                frames.append(piece_cells(move.piece, move.orientation, x, y))
        self.animation = frames
        self.anim_timer.start()
        self._animate()

    def _animate(self) -> None:
        if not self.animation:
            self.anim_timer.stop()
            self._render()
            return
        cells = self.animation.pop(0)
        move = self.results[self.selected].move if self.results else None
        color = COLORS[PIECES[move.piece]] if move else "#ffffff"
        self.board.show_state(self.state.field, cells, color)

    def request_build(self) -> None:
        self.build_button.hide()
        self.instruction.setText("Liczę tabele dla tego stanu worka…")
        self.build_requested.emit(self.pending_offset)

    def refresh(self) -> None:
        self.build_button.hide()
        self.anim_timer.stop()
        self.animation = []
        self.results = []
        self.selected = 0
        self.request += 1
        state = self.state
        self.move_spin.blockSignals(True)
        self.move_spin.setValue(min(state.t + 1, TOTAL_PIECES))
        self.move_spin.blockSignals(False)
        self._render_controls()
        self._render()
        if state.t >= TOTAL_PIECES:
            done = state.field == 0
            self.instruction.setText("Level zaliczony: plansza pusta!" if done else "Koniec ruchów, plansza nie jest pusta.")
            return
        if state.current is None:
            self.instruction.setText("Wybierz klocek, który teraz spada.")
            return
        lines = Solver.lines_remaining(state.field, state.t)
        if lines is None or lines <= 0:
            self.instruction.setText("Liczba kostek nie pasuje do numeru ruchu. Popraw planszę albo ruch.")
            return
        offset = Solver.offset_for(state.t, state.bag_after_current)
        if not self.solver.values_ready(offset):
            self.pending_offset = offset
            self.instruction.setText("Ten stan worka wymaga osobnych tabel. Sprawdź worek albo policz je przyciskiem.")
            self.build_button.show()
            return
        self.instruction.setText("Liczę…")
        task = AnalyzeTask(self.request, self.solver, state)
        task.signals.finished.connect(self.on_analyzed)
        self.pool.start(task)

    def on_analyzed(self, request: int, results: list, elapsed: float, error: str) -> None:
        if request != self.request:
            return
        self.results = results
        self.selected = 0
        if error:
            self.instruction.setText(error)
        elif not results or results[0].probability <= 0:
            self.instruction.setText("Nie da się – restartuj level.")
        self.status.emit(f"Ostatnie liczenie: {elapsed * 1000:.0f} ms")
        self._render()

    def _render_controls(self) -> None:
        state = self.state
        for i, button in enumerate(self.current_buttons[1]):
            button.setChecked(state.current == i)
            button.setEnabled(bool((state.pool >> i) & 1))
        pool_next = (state.bag_after_current or FULL_BAG) if state.current is not None else 0
        for i, button in enumerate(self.next_buttons[1]):
            button.setChecked(state.nxt == i)
            button.setEnabled(bool((pool_next >> i) & 1))
        if state.current is None:
            shown = state.pool
        else:
            shown = state.bag_after_current
            if state.nxt is not None:
                shown = remove_piece(shown, state.nxt)
        for i, button in enumerate(self.bag_buttons[1]):
            button.setChecked(bool((shown >> i) & 1))
        cleared = state.lines_cleared
        cleared_text = "?" if cleared is None else str(cleared)
        lines = Solver.lines_remaining(state.field, min(state.t, TOTAL_PIECES))
        block = "–" if lines is None else str(min(lines, 4))
        self.counters.setText(
            f"Ruch <b>{min(state.t + 1, TOTAL_PIECES)}</b>/{TOTAL_PIECES} &nbsp; "
            f"Linie <b>{cleared_text}</b>/{TOTAL_LINES} &nbsp; Blok (cel L) <b>{block}</b>"
        )
        self.undo_button.setEnabled(bool(self.history))

    def _render(self) -> None:
        state = self.state
        for i, button in enumerate(self.move_buttons):
            if i < len(self.results) and self.results[i].probability > 0:
                evaluation = self.results[i]
                button.setText(f"{i + 1}.  {percent(evaluation.probability)}   {describe_move(evaluation.move, self.config['rotation_tap'])[3:]}")
                button.setEnabled(True)
                button.setChecked(i == self.selected)
                button.setToolTip(describe_move(evaluation.move, self.config["rotation_tap"]))
            else:
                button.setText("–")
                button.setEnabled(False)
                button.setChecked(False)
        playable = bool(self.results) and self.results[self.selected].probability > 0
        self.placed.setEnabled(playable)
        self.path_button.setEnabled(playable)
        if playable:
            evaluation = self.results[self.selected]
            move = evaluation.move
            self.board.show_state(state.field, move.cells, COLORS[PIECES[move.piece]])
            self.instruction.setText(f"{describe_move(move, self.config['rotation_tap'])}  →  {percent(evaluation.probability)}")
        else:
            self.board.show_state(state.field, [], "#ffffff")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tetris PC Solver – level 602")
        self.resize(1040, 800)
        self.config_data = load_config()
        self.solver = Solver()
        self._remove_interrupted_builds()
        QThreadPool.globalInstance().start(_warm_up)
        self.process: QProcess | None = None
        self.progress_path: Path | None = None
        self.progress_position = 0
        self.build_timer = QTimer(self)
        self.build_timer.setInterval(500)
        self.build_timer.timeout.connect(self.read_build_progress)
        self.game = GameTab(self.solver, self.config_data)
        self.simulation = SimulationTab(self.solver)
        tabs = QTabWidget()
        tabs.addTab(self.game, "Gra")
        tabs.addTab(self.simulation, "Symulacja")
        self.setCentralWidget(tabs)
        self.cache_label = QLabel()
        self.time_label = QLabel()
        self.build_bar = QProgressBar()
        self.build_bar.setFixedWidth(280)
        self.build_bar.setRange(0, 1000)
        self.build_bar.hide()
        self.statusBar().addWidget(self.cache_label)
        self.statusBar().addWidget(self.build_bar)
        self.statusBar().addPermanentWidget(self.time_label)
        self.game.status.connect(self.time_label.setText)
        self.game.build_requested.connect(self.start_build)
        self.update_cache_label()
        if not self.solver.values_ready(0):
            self.start_build(0)

    def _remove_interrupted_builds(self) -> None:
        """Usuwa pliki tymczasowe; plik używany przez inny proces zostaje."""
        for pattern in ("*.tmp", "*.tmp.npy", "*.tmp.npz", "*.progress"):
            for path in self.solver.cache.glob(pattern):
                try:
                    path.unlink()
                except OSError:
                    pass

    def update_cache_label(self) -> None:
        ready = [offset for offset in range(7) if self.solver.values_ready(offset)]
        if 0 in ready:
            text = f"Tabele gotowe (P startu: {self.solver.level_value() * 100:.2f}%)".replace(".", ",")
        elif ready:
            text = "Tabele gotowe dla innego stanu worka"
        else:
            text = "Brak tabel, trwa pierwsze liczenie"
        self.cache_label.setText(text)

    def start_build(self, offset: int) -> None:
        if self.process is not None:
            return
        self.solver.cache.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.solver.cache / f"build_{offset}.progress"
        self.progress_path.unlink(missing_ok=True)
        self.progress_position = 0
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(APP_DIR))
        self.process.finished.connect(self.build_finished)
        self.build_bar.setValue(0)
        self.build_bar.show()
        self.cache_label.setText("Liczenie tabel…")
        arguments = ["--build-tables", str(offset), str(self.progress_path)]
        if not getattr(sys, "frozen", False):
            arguments.insert(0, str(Path(__file__).resolve()))
        self.process.start(sys.executable, arguments)
        self.build_timer.start()

    def read_build_progress(self) -> None:
        try:
            with open(self.progress_path, "rb") as handle:
                handle.seek(self.progress_position)
                data = handle.read()
        except OSError:
            return
        end = data.rfind(b"\n")
        if end < 0:
            return
        self.progress_position += end + 1
        fraction, message = data[:end].decode("utf-8").splitlines()[-1].split(" ", 1)
        self.build_bar.setValue(int(float(fraction) * 1000))
        self.cache_label.setText(message)

    def build_finished(self, exit_code: int, _status) -> None:
        self.build_timer.stop()
        self.progress_path.unlink(missing_ok=True)
        self.process.deleteLater()
        self.process = None
        self.build_bar.hide()
        self.update_cache_label()
        if exit_code != 0:
            self.cache_label.setText("Liczenie tabel nie powiodło się")
            return
        self.game.refresh()

    def closeEvent(self, event) -> None:
        if self.process is not None:
            self.process.kill()
            self.process.waitForFinished(2000)
        worker = self.simulation.thread_worker
        if worker is not None:
            worker.cancelled = True
            worker.wait(5000)
        super().closeEvent(event)


def _warm_up() -> None:
    """Kompiluje z góry funkcję używaną przy pierwszej podpowiedzi."""
    clear_rows(0, 4)


def run_self_check(report_path: str) -> None:
    """Sprawdza spakowaną aplikację bez okna i zapisuje wynik do pliku."""
    solver = Solver()
    current = PIECES.index("T")
    bag = remove_piece(FULL_BAG, current)
    started = time.perf_counter()
    best = solver.analyze(0, 0, current, PIECES.index("I"), bag)[0]
    first = time.perf_counter() - started
    started = time.perf_counter()
    solver.analyze(0, 0, current, PIECES.index("O"), bag)
    second = time.perf_counter() - started
    result = simulate("optimal", 100_000, solver=solver)
    Path(report_path).write_text(
        f"start={solver.level_value() * 100:.3f}% best={best.probability * 100:.3f}% "
        f"first_ms={first * 1000:.0f} next_ms={second * 1000:.1f} "
        f"sim={result.win_rate * 100:.3f}% +-{result.margin * 100:.3f} threads={result.threads}\n",
        encoding="utf-8",
    )


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--build-tables":
        Solver().ensure(int(sys.argv[2]), file_progress(sys.argv[3]))
        return
    if len(sys.argv) == 3 and sys.argv[1] == "--self-check":
        run_self_check(sys.argv[2])
        return
    application = QApplication(sys.argv)
    application.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    sys.exit(application.exec())


if __name__ == "__main__":
    main()
