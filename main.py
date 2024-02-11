import csv
import decimal
import io
import math
import sys
import cv2
import copy
import keyboard
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtWidgets import (QApplication, QFileDialog, QMainWindow, QGraphicsPixmapItem, QGraphicsPolygonItem, QDialog,
                             QMessageBox, QGraphicsScene)
from PyQt5.QtGui import QPixmap, QPolygonF, QPen, QColor, QImage, QKeyEvent, QBrush

# UI side imports
from ui.main_window_ui import Ui_MainWindow as MainWindowUI
from ui.filter_ui import Ui_Dialog as FilterDialogUI

# Pipeline side imports
print('Loading libraries... Please wait')
#import scripts.evaluate as evaluate
import scripts.find_cells as find_cells
from scripts.measure import get_all_metric_names
print('Finished loading')

network_strategy_file = Path('./model/D13_BEST_20220717.strat')
network_weights_file = Path('./model/D13_BEST_20220717_weights.h5')
network_trained_cell_erosion = 2

KEY_SHOW_ALL_CELLS = Qt.Key_Space


class Cell:
    def __init__(self, cell_id: int, data: tuple):
        self.id = cell_id
        self.parameters, self.contour_normalized, self.contour_square_fit = data


class Filter:
    def __init__(self, parameter_name: str, active: bool = False):
        self._parameter_name = parameter_name
        self._bounds = [None, None]
        self._active = active

    def __call__(self, cell, nan_allow_pass=False) -> bool:
        if not self._active:
            return True

        parameter_value = cell.parameters[self.get_name()]

        if np.isfinite(parameter_value):
            if self.has_lower_bound() and parameter_value < self.get_lower_bound():
                return False
            if self.has_upper_bound() and parameter_value > self.get_upper_bound():
                return False
            return True
        else:
            return nan_allow_pass

    def __str__(self):
        return f'Filter for parameter {self._parameter_name}. Bounds ({self._bounds[0]}, {self._bounds[1]}). Filter is active={self._active}'

    def has_lower_bound(self) -> bool:
        return self._bounds[0] is not None

    def has_upper_bound(self) -> bool:
        return self._bounds[1] is not None

    def get_name(self) -> str:
        return self._parameter_name

    def get_lower_bound(self) -> float:
        return self._bounds[0]

    def get_upper_bound(self) -> float:
        return self._bounds[1]

    def set_active(self, active: bool):
        self._active = active

    def set_lower_bound(self, value: float):
        self._bounds[0] = value

    def set_upper_bound(self, value: float):
        self._bounds[1] = value


class CellRegistry:
    def __init__(self, list_of_cells=None):
        if list_of_cells is None:
            list_of_cells = []
        self.cells = dict()
        self.statistics = []
        self.active_cells = set()
        self.filtered_cells = set()
        self.manual_inclusions = set()
        self.manual_exclusions = set()
        for cell_id, cell_data in enumerate(list_of_cells):
            self.cells[cell_id] = Cell(cell_id, cell_data)
            self.active_cells.add(cell_id)
            self.filtered_cells.add(cell_id)

        self.parameter_names = get_all_metric_names()
        self.parameter_ranges = {name: [None, None] for name in self.parameter_names}
        for cell_id, cell in self.cells.items():
            for name in self.parameter_names:
                entry = self.parameter_ranges[name]
                value = cell.parameters[name]
                if entry[0] is None or entry[0] > value:
                    entry[0] = value
                if entry[1] is None or entry[1] < value:
                    entry[1] = value
                self.parameter_ranges[name] = entry

        self.filters = []
        self.build_filters()

    def generate_cell_parameter_text(self, cell_id: int) -> str:
        if cell_id not in self.cells:
            return f'Cell with id {cell_id} was not found in registry'
        else:
            cell = self.cells[cell_id]
            if self.is_cell_active(cell_id):
                text = f'Parameters of cell {cell_id} (included):'
            else:
                text = f'Parameters of cell {cell_id} (excluded):'
            with decimal.localcontext() as ctx:
                ctx.prec = 5
                for parameter_name, value in cell.parameters.items():
                    if isinstance(value, bool):
                        text += f'\n- {parameter_name}: {value}'
                    else:
                        value = ctx.create_decimal(value)
                        text += f'\n- {parameter_name}: {value}'
            return text

    def generate_parameter_statistics_text(self) -> str:
        self.compute_statistics()

        if len(self.statistics) == 0:
            return 'No statistics to show.'
        else:
            text = f'Statistics for a total of {len(self.get_active_cells())} active cells:'
            for stat in self.statistics:
                name, parameters = stat
                with decimal.localcontext() as ctx:
                    ctx.prec = 5
                    mean = ctx.create_decimal(parameters['mean'])
                    sem = ctx.create_decimal(parameters['sem'])
                    minimum = ctx.create_decimal(parameters['min'])
                    maximum = ctx.create_decimal(parameters['max'])
                    text += f'\n- {name}: {mean} [{sem}]; (min/max) ({minimum}/{maximum})'
            return text

    def export_to_file(self, filepath):
        with open(str(filepath), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['cell_id'] + self.parameter_names)
            for cell_id in self.get_active_cells():
                params = self.get_cell(cell_id).parameters
                writer.writerow([cell_id] + [params[name] for name in self.parameter_names])

    def build_filters(self):
        self.filters.clear()
        for parameter_name in self.parameter_names:
            if parameter_name != 'Is edge cell':
                self.filters.append(Filter(parameter_name, active=True))

    def check_against_filters(self, cell: Cell):
        for f in self.filters:
            if not f(cell):
                break
        else:
            return True
        return False

    def compute_statistics(self):
        self.statistics.clear()
        if len(self.get_active_cells()) == 0:
            return

        for parameter_name in self.parameter_names:
            if parameter_name == 'Is edge cell':
                continue  # Skip this boolean parameter

            array = self.get_data_for_parameter(parameter_name)
            self.statistics.append((parameter_name, {'min': array.min(), 'max': array.max(), 'mean': array.mean(),
                                                     'median': np.median(array), 'stddev': array.std(),
                                                     'sem': array.std() / np.sqrt(array.shape[0])}))

    def update_filtering(self):
        self.active_cells.clear()
        self.filtered_cells.clear()
        for cell_id, cell in self.cells.items():
            if self.check_against_filters(cell):
                self.filtered_cells.add(cell_id)

        self.active_cells = self.filtered_cells.union(self.manual_inclusions).difference(self.manual_exclusions)

    def get_filter(self, index: int) -> Filter:
        return self.filters[index]

    def get_filters(self) -> list:
        return self.filters

    def set_filters(self, filters: list):
        self.filters = filters

    def get_active_cells(self) -> list:
        return list(self.active_cells)

    def get_num_cells_to_draw(self) -> int:
        return len(self.cells)

    def get_cells_to_draw(self) -> list:
        return list(self.cells.keys())

    def get_cell(self, cell_id: int) -> Cell:
        return self.cells[cell_id]

    def get_parameter_range(self, name: str) -> [float, float]:
        return self.parameter_ranges[name].copy()

    def get_data_for_parameter(self, name: str) -> np.ndarray:
        array = []
        for cell_id in self.get_active_cells():
            array.append(self.get_cell(cell_id).parameters[name])
        return np.array(array)

    def is_cell_active(self, cell_id: int) -> bool:
        return cell_id in self.active_cells

    def reset_manual_filtering(self):
        self.manual_inclusions.clear()
        self.manual_exclusions.clear()
        self.update_filtering()

    def toggle_statistics_inclusion(self, cell_id: int) -> bool:
        if cell_id in self.active_cells:
            # Cell is active, so deactivate
            self.active_cells.remove(cell_id)

            if self.check_against_filters(self.get_cell(cell_id)):
                # Cell passes filters, so forcefully exclude
                self.manual_exclusions.add(cell_id)
            else:
                # Cell does not pass filters, but is active, so remove from forceful inclusion list
                self.manual_inclusions.remove(cell_id)

            return False
        else:
            # Cell is inactive, so activate
            self.active_cells.add(cell_id)

            if self.check_against_filters(self.get_cell(cell_id)):
                # Cell passes filters, but is not active, so remove from forceful exclusion list
                self.manual_exclusions.remove(cell_id)
            else:
                # Cell does not pass filters, so forcefully include
                self.manual_inclusions.add(cell_id)

            return True


class ClickableGraphicsScene(QGraphicsScene):
    def __init__(self, parent, click_callback=None, double_click_callback=None):
        super().__init__(parent)
        self.my_click_callback = click_callback
        self.my_double_click_callback = double_click_callback

    def mouseReleaseEvent(self, event) -> None:
        if self.my_click_callback:
            self.my_click_callback(self, event)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if self.my_double_click_callback:
            self.my_double_click_callback(self, event)
        super().mouseDoubleClickEvent(event)


def get_drawable_rect_size(graphics_view) -> (int, int):
    rect = graphics_view.frameRect()
    return rect.width() - 2 * graphics_view.lineWidth(), rect.height() - 2 * graphics_view.lineWidth()


def render_cell(cell, width: int, height: int):
    points = cell.contour_square_fit

    def to_pixel_space(point):
        return round(point[0] * width), round(point[1] * height)

    poly = np.array(list(map(to_pixel_space, points)), dtype=np.int32)
    image = cv2.fillPoly(np.zeros(shape=(height, width, 1), dtype=np.uint8), poly[np.newaxis, :, :], color=255)

    image = QImage(image, width, height, width, QImage.Format_Grayscale8)
    return QPixmap(image)


empty_image = None


def get_empty_pixmap(width: int, height: int, color: int = 0):
    global empty_image
    if empty_image is None:
        empty_image = QImage(np.array([color], dtype=np.uint8), 1, 1, 1, QImage.Format_Grayscale8)
    return QPixmap(empty_image).scaled(width, height, 0, 1)


class Page:
    def __init__(self, cell_ids: list, cell_registry):
        self.cell_ids = cell_ids
        self.cell_registry = cell_registry
        self.selected_cell_view_index = -1

    def render(self, graphics_views: list, graphics_pixmap_items: list, graphics_selected_rect_items: list,
               graphics_signal_rect_items: list):
        for index in range(len(graphics_pixmap_items)):
            width, height = get_drawable_rect_size(graphics_views[index])
            if index < len(self.cell_ids):
                cell = self.cell_registry.get_cell(self.cell_ids[index])
                pixmap = render_cell(cell, width, height)
                graphics_pixmap_items[index].setPixmap(pixmap)
                graphics_selected_rect_items[index].setVisible(index == self.selected_cell_view_index)
                graphics_signal_rect_items[index].setVisible(
                    not self.cell_registry.is_cell_active(self.cell_ids[index]))
            else:
                graphics_pixmap_items[index].setPixmap(get_empty_pixmap(width, height))
                graphics_selected_rect_items[index].setVisible(False)
                graphics_signal_rect_items[index].setVisible(False)

    def set_cell_view_selected(self, index: int) -> int:
        if 0 <= index < len(self.cell_ids):
            self.selected_cell_view_index = index
            return self.cell_ids[index]
        else:
            self.selected_cell_view_index = -1
            return -1


class PageManager:
    def __init__(self, rows: int, cols: int, main_window, cell_registry, show_hidden_by_filter: bool, sorter):
        self.rows = rows
        self.cols = cols
        self.main_window = main_window
        self.cell_registry = cell_registry
        self.show_hidden_by_filter = show_hidden_by_filter
        self.current_page = 0
        self.sorter = sorter
        self.sorted_cell_ids = sorter(self.cell_registry)
        self.pages = []
        self._build_pages()
        self.graphics_views = []
        self.graphics_pixmap_items = []
        self.graphics_selected_rect_items = []  # Red boxes for showing selection
        self.graphics_signal_rect_items = []  # Overlay rects for showing statistics inclusion/exclusion
        self.selected_cell_id = -1
        self.selected_cell_view_index = -1

    def get_ui_state(self):
        if len(self.pages) > 0:
            page_label_text = f'Page {self.current_page + 1}/{len(self.pages)}'
        else:
            page_label_text = 'Page -/-'
        return {'button_page_next_enabled': self.current_page < len(self.pages) - 1,
                'button_page_prev_enabled': self.current_page > 0,
                'cell_selected': self.has_selected_cell(),
                'page_label_text': page_label_text}

    def has_selected_cell(self) -> bool:
        return self.selected_cell_view_index >= 0

    def get_selected_cell_id(self) -> int:
        return self.selected_cell_id

    def change_page_prev(self):
        if self.current_page > 0:
            self.pages[self.current_page].set_cell_view_selected(-1)
            self.current_page -= 1
            self.selected_cell_id = -1
            self.selected_cell_view_index = -1
            # self.main_window.highlight_cell(self.selected_cell_id)
            self.main_window.update()
        else:
            print('Cannot change to previous page. Already at first page!')

    def change_page_next(self):
        if self.current_page < len(self.pages) - 1:
            self.pages[self.current_page].set_cell_view_selected(-1)
            self.current_page += 1
            self.selected_cell_id = -1
            self.selected_cell_view_index = -1
            # self.main_window.highlight_cell(self.selected_cell_id)
            self.main_window.update()
        else:
            print('Cannot change to next page. Already at last page!')

    def render(self):
        if len(self.sorted_cell_ids) > 0:
            self.pages[self.current_page].render(self.graphics_views, self.graphics_pixmap_items,
                                                 self.graphics_selected_rect_items, self.graphics_signal_rect_items)
        else:
            for index, item in enumerate(self.graphics_pixmap_items):
                width, height = get_drawable_rect_size(self.graphics_views[index])
                item.setPixmap(get_empty_pixmap(width, height))

    def connect_graphics_views(self, graphics_views: list):
        self.graphics_views = graphics_views
        self.graphics_pixmap_items.clear()
        self.graphics_selected_rect_items.clear()
        for index in range(len(self.graphics_views)):
            callback = self._make_click_callback(index)
            scene = ClickableGraphicsScene(self.main_window, click_callback=callback)
            width, height = get_drawable_rect_size(self.graphics_views[index])
            pixmap = get_empty_pixmap(width, height)
            item = QGraphicsPixmapItem(pixmap)
            self.graphics_pixmap_items.append(item)
            scene.addItem(item)

            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(2)
            border_dist = 2
            item = scene.addRect(border_dist, border_dist, width - 2 * border_dist, height - 2 * border_dist, pen)
            item.setVisible(False)
            self.graphics_selected_rect_items.append(item)

            pen = QPen(QColor(255, 0, 0))
            brush = QBrush(QColor(128, 128, 128), Qt.BDiagPattern)
            item = scene.addRect(-1, -1, width + 1, height + 1, pen, brush)
            item.setVisible(False)
            self.graphics_signal_rect_items.append(item)

            self.graphics_views[index].setScene(scene)
            self.graphics_views[index].setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.graphics_views[index].setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def deselect_cell(self):
        self.pages[self.current_page].set_cell_view_selected(-1)
        self.selected_cell_view_index = -1
        self.selected_cell_id = -1

    def seek_to_cell(self, cell_id: int):
        for page_index, page in enumerate(self.pages):
            try:
                self.pages[self.current_page].set_cell_view_selected(-1)
                index_on_page = page.cell_ids.index(cell_id)
                self.current_page = page_index
                self.selected_cell_id = cell_id
                self.selected_cell_view_index = index_on_page
                self.pages[self.current_page].set_cell_view_selected(self.selected_cell_view_index)
                break
            except ValueError:
                pass
        else:
            print(f'Could not seek to cell with id {cell_id} as it does not exist')

    def switch_cell_registry(self, new_cr):
        self.cell_registry = new_cr
        self.resort(self.sorter)

    def resort(self, sorter):
        self.sorter = sorter
        self.reset()

    def reset(self):
        self.sorted_cell_ids = self.sorter(self.cell_registry)
        self._build_pages()
        self.current_page = 0
        self.selected_cell_id = -1
        self.selected_cell_view_index = -1
        if len(self.pages) > 0:
            self.pages[self.current_page].set_cell_view_selected(self.selected_cell_view_index)
        # self.main_window.highlight_cell(self.selected_cell_id)
        self.main_window.update()

    def _make_click_callback(self, index):
        page_man = self

        def on_clicked_cell_view(widget, event):
            if len(page_man.pages) == 0:
                return
            print('Clicked on cell view with index', index)
            if index != page_man.selected_cell_view_index:
                print('Setting selected cell to', index)
                page_man.selected_cell_view_index = index
            else:
                print('Deselecting cell')
                page_man.selected_cell_view_index = -1
            cell_id = page_man.pages[page_man.current_page].set_cell_view_selected(page_man.selected_cell_view_index)
            page_man.selected_cell_id = cell_id
            if cell_id >= 0:
                print(f'Current selected cell_id is {page_man.selected_cell_id}')
            else:
                print('Clicked on empty cell view, Deselecting current cell...')
                page_man.selected_cell_view_index = -1

            # page_man.main_window.highlight_cell(cell_id)
            page_man.main_window.check_show_all_cells()
            page_man.main_window.update()

        return on_clicked_cell_view

    def _build_pages(self):
        per_page = self.rows * self.cols
        num_pages = math.ceil(float(len(self.sorted_cell_ids)) / per_page)
        self.pages.clear()
        for index in range(num_pages):
            page = Page(self.sorted_cell_ids[index * per_page:(index + 1) * per_page], self.cell_registry)
            self.pages.append(page)


def default_sorter(cell_registry):
    return cell_registry.get_cells_to_draw()


class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = MainWindowUI()
        self.ui.setupUi(self)

        self.displays_something = False
        self.show_all_cells = True
        self.adjusting_filter = False
        self.cell_views = [self.ui.cell_view_1, self.ui.cell_view_2, self.ui.cell_view_3,
                           self.ui.cell_view_4, self.ui.cell_view_5, self.ui.cell_view_6,
                           self.ui.cell_view_7, self.ui.cell_view_8, self.ui.cell_view_9]
        self.prediction = None
        self.input_image = None
        self.input_image_drawn_size = None

        self.connect_signals_slots()
        self.show()  # Force layouts to compute widget rects

        self.cell_registry = CellRegistry()
        self.page_manager = PageManager(3, 3, self, self.cell_registry, False, default_sorter)
        self.page_manager.connect_graphics_views(self.cell_views)

        self.input_image_pixmap_container = None
        self.input_image_view_scene = None
        self.init_input_image_view()
        self.input_image_view_cell_polygon_items = dict()

        self.sorter_combo_box_to_sorter_fn = []
        self.build_sorting_combo_box()
        self.on_choose_sorter(0)  # Init to area desc. sorter

        self.update()

    def update(self):
        pm_ui_state = self.page_manager.get_ui_state()
        self.ui.button_page_next.setEnabled(pm_ui_state['button_page_next_enabled'])
        self.ui.button_page_prev.setEnabled(pm_ui_state['button_page_prev_enabled'])
        self.ui.button_toggle_inclusion.setEnabled(pm_ui_state['cell_selected'])
        # self.ui.button_edit_cell.setEnabled(pm_ui_state['cell_selected'])
        self.ui.text_current_cell.setEnabled(pm_ui_state['cell_selected'])
        self.ui.label_current_page.setText(pm_ui_state['page_label_text'])
        self.page_manager.render()

        # Draw cells onto input image
        for cid, polygon_item in self.input_image_view_cell_polygon_items.items():
            if self.adjusting_filter:
                if self.cell_registry.is_cell_active(cid):
                    pen = QPen(QColor(0, 255, 0))
                    polygon_item.setPen(pen)
                    polygon_item.setVisible(True)
                else:
                    pen = QPen(QColor(255, 0, 0))
                    polygon_item.setPen(pen)
                    polygon_item.setVisible(True)
            else:
                if cid == self.page_manager.get_selected_cell_id():
                    pen = QPen(QColor(255, 0, 0))
                    polygon_item.setPen(pen)
                    polygon_item.setVisible(True)
                else:
                    pen = QPen(QColor(255, 0, 0))
                    polygon_item.setPen(pen)
                    polygon_item.setVisible(self.show_all_cells)

        self.ui.button_reset_filters.setEnabled(self.displays_something)
        self.ui.button_edit_filters.setEnabled(self.displays_something)
        # self.ui.button_edit_segmentation.setEnabled(self.displays_something)
        self.ui.button_reset_manual_filtering.setEnabled(self.displays_something)
        self.ui.combo_sort_by.setEnabled(self.displays_something)
        self.ui.text_statistics.setEnabled(self.displays_something)

        self.draw_statistics()

    def connect_signals_slots(self):
        # Menu
        self.ui.actionLoad_image.triggered.connect(self.load_image)
        self.ui.actionImport_segmentation.triggered.connect(self.import_segmentation)
        self.ui.actionExport_data_2.triggered.connect(self.export_data)
        self.ui.actionExport_segmentation.triggered.connect(self.export_segmentation)
        self.ui.actionExport_annotated_image.triggered.connect(self.export_annotated_image)
        self.ui.actionForce_update.triggered.connect(self.update)

        # Buttons
        self.ui.button_page_next.clicked.connect(self.next_page)
        self.ui.button_page_prev.clicked.connect(self.prev_page)
        self.ui.button_toggle_inclusion.clicked.connect(self.on_toggle_statistics_inclusion)
        self.ui.button_reset_manual_filtering.clicked.connect(self.on_reset_manual_filtering)
        self.ui.button_edit_filters.clicked.connect(self.on_edit_filters)
        self.ui.button_reset_filters.clicked.connect(self.on_reset_filters)

        # Combo boxes
        self.ui.combo_sort_by.activated.connect(self.on_choose_sorter)

    def _process_prediction(self):
        print(f'Extracting cells from prediction...')
        data = find_cells.find_cells(self.prediction, network_trained_cell_erosion)

        print(f'Found cells, prepare for visualization')
        self.cell_registry = CellRegistry(data)
        self.page_manager.switch_cell_registry(self.cell_registry)

        self.build_cell_polygons()
        self.displays_something = True
        self.show_all_cells = True
        self.update()

    def load_image(self):
        try:
            print('Selecting image file to load...')
            filepath, _ = QFileDialog.getOpenFileName(self, 'Load input image', '', 'All Files (*)')

            if filepath == '':
                print('Aborted load image')
                return

            filepath = Path(filepath)

            if filepath.exists() and filepath.is_file():
                print(f'Selected input file: {filepath}')
            else:
                print(f'Given file does not exist: {filepath}')
                return

            width, height = get_drawable_rect_size(self.ui.input_image_view)
            self.input_image = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
            if self.input_image.shape[0] > self.input_image.shape[1]:
                width = int(np.round(height * self.input_image.shape[1] / self.input_image.shape[0]))
            else:
                height = int(np.round(width * self.input_image.shape[0] / self.input_image.shape[1]))
            self.input_image_drawn_size = (width, height)

            pixmap = QPixmap(str(filepath)).scaled(width, height, 0, 1)
            self.input_image_pixmap_container.setPixmap(pixmap)

            print(f'Processing in neural network...')
            #self.prediction = evaluate.run_on_single_from_weights(network_strategy_file, network_weights_file, filepath,
            #                                                      noise_level=0.0)
            self.prediction = None
            self._process_prediction()
        except Exception as e:
            print('Could not process image as an exception occurred:')
            print(e)
            print('Recovering...')

            width, height = get_drawable_rect_size(self.ui.input_image_view)
            pixmap = get_empty_pixmap(width, height)
            self.input_image_pixmap_container.setPixmap(pixmap)

            self.cell_registry = CellRegistry()
            self.page_manager = PageManager(3, 3, self, self.cell_registry, False, default_sorter)
            self.page_manager.connect_graphics_views(self.cell_views)

    def export_data(self):
        if not self.displays_something:
            QMessageBox.question(self, 'Export data...', 'Nothing to export right now!', QMessageBox.Ok)
            return

        filepath, _ = QFileDialog.getSaveFileName(self, 'Export to file', '',
                                                  'All Files (*);;Comma-separated file (*.csv)',
                                                  'Comma-separated file (*.csv)')

        if filepath == '':
            print('Aborted export')
            return

        if Path(filepath).parent.exists():
            print(f'Selected output file: {filepath}')
        else:
            print(f'Path to file does not exist: {filepath}')
            return

        print('Exporting data...')
        self.cell_registry.export_to_file(filepath)
        print('Finished export')

    def import_segmentation(self):
        if not self.displays_something:
            QMessageBox.question(self, 'Import segmentation...', 'Load an image first!', QMessageBox.Ok)
            return

        filepath, _ = QFileDialog.getOpenFileName(self, 'Import segmentation image', '',
                                                  'Bitmap Image (*.bmp);;Portable Network Graphics (*.png)')
        if filepath == '':
            print('Aborted import of segmentation')
            return

        filepath = Path(filepath)

        try:
            image = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
            cv2.imshow('Chosen segmentation image', image)
        except Exception as e:
            print('Error occurred while importing segmentation image:', filepath)
            print(e)
            return

        choice = QMessageBox.question(self, 'Import segmentation...',
                                      'Use the shown segmentation?',
                                      QMessageBox.Yes | QMessageBox.No)
        cv2.destroyWindow('Chosen segmentation image')
        if choice == QMessageBox.No:
            print('Aborting import of segmentation')
            return

        print('Loading segmentation from file:', filepath)
        self.prediction = image / 255.0
        self._process_prediction()

    def export_segmentation(self):
        if not self.displays_something or self.prediction is None:
            QMessageBox.question(self, 'Export segmentation...', 'Nothing to export right now!', QMessageBox.Ok)
            return

        filepath, _ = QFileDialog.getSaveFileName(self, 'Export segmentation image', '',
                                                  'All Files (*);;Bitmap Image (*.bmp)', 'Bitmap Image (*.bmp)')
        if filepath == '':
            print('Aborted export of segmentation')
            return

        filepath = Path(filepath)
        filepath.parent.mkdir(exist_ok=True, parents=True)

        cv2.imwrite(str(filepath), (self.prediction * 255.0).astype(np.uint8))

        print('Exported segmentation to file:', filepath)

    def export_annotated_image(self):
        if not self.displays_something or self.prediction is None:
            QMessageBox.question(self, 'Export annotated image...', 'Nothing to export right now!', QMessageBox.Ok)
            return

        filepath, _ = QFileDialog.getSaveFileName(self, 'Export annotated image', '',
                                                  'All Files (*);;Bitmap Image (*.bmp)', 'Bitmap Image (*.bmp)')
        if filepath == '':
            print('Aborted export of annotated image')
            return

        filepath = Path(filepath)
        filepath.parent.mkdir(exist_ok=True, parents=True)

        contours = [self.cell_registry.get_cell(id).contour_normalized for id in self.cell_registry.get_active_cells()]
        annotated = find_cells.draw_cells_to_image(self.input_image, contours, color=(255, 0, 0))

        cv2.imwrite(str(filepath), annotated)

        print('Exported annotated image to file:', filepath)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_H and not self.show_all_cells:
            self.show_all_cells = True
            self.update()
        event.accept()

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_H and not keyboard.is_pressed('h'):
            self.show_all_cells = False
            self.update()
        event.accept()

    def check_show_all_cells(self):
        self.show_all_cells = keyboard.is_pressed(' ')

    def on_click_input_view(self, widget, event):
        print('Clicked into input image view at', event.scenePos())

        point = event.scenePos()
        for cell_id, polygon_item in self.input_image_view_cell_polygon_items.items():
            if polygon_item.contains(point):
                self.select_cell(cell_id)
                return

    def on_toggle_statistics_inclusion(self):
        if self.page_manager.has_selected_cell():
            cell_id = self.page_manager.get_selected_cell_id()
            self.cell_registry.toggle_statistics_inclusion(cell_id)
        self.update()

    def on_reset_manual_filtering(self):
        self.cell_registry.reset_manual_filtering()
        self.update()

    def on_edit_filters(self):
        self.adjusting_filter = True
        dialog = FilterDialog(self, self.cell_registry)
        dialog.setModal(True)
        if dialog.exec_() == 1:
            self.cell_registry.update_filtering()
            self.page_manager.reset()
        self.adjusting_filter = False
        self.update()

    def on_reset_filters(self):
        self.cell_registry.build_filters()
        self.cell_registry.update_filtering()
        self.update()

    def on_choose_sorter(self, index: int):
        sorter = self.sorter_combo_box_to_sorter_fn[index]
        self.page_manager.resort(sorter)

    def next_page(self):
        self.page_manager.change_page_next()
        self.update()

    def prev_page(self):
        self.page_manager.change_page_prev()
        self.update()

    def select_cell(self, cell_id: int):
        if self.page_manager.selected_cell_id != cell_id:
            print(f'Selecting cell {cell_id}')
            self.page_manager.seek_to_cell(cell_id)
        else:
            print(f'Deselecting cell {cell_id}')
            self.page_manager.deselect_cell()
        self.check_show_all_cells()
        self.update()

    def draw_statistics(self):
        self.ui.text_current_cell.setReadOnly(True)
        if self.page_manager.has_selected_cell():
            self.ui.label_current_cell.setText(f'Selected cell id: {self.page_manager.get_selected_cell_id()}')
            text = self.cell_registry.generate_cell_parameter_text(self.page_manager.get_selected_cell_id())
            self.ui.text_current_cell.setPlainText(text)
        else:
            self.ui.label_current_cell.setText('Selected cell id: -')
            self.ui.text_current_cell.setPlainText('No cell selected.')

        self.ui.text_statistics.setReadOnly(True)
        self.ui.text_statistics.setPlainText(self.cell_registry.generate_parameter_statistics_text())

    def init_input_image_view(self):
        self.input_image_view_scene = ClickableGraphicsScene(self)
        width, height = get_drawable_rect_size(self.ui.input_image_view)
        pixmap = get_empty_pixmap(width, height)
        self.input_image_pixmap_container = QGraphicsPixmapItem(pixmap)
        self.input_image_view_scene.addItem(self.input_image_pixmap_container)
        self.ui.input_image_view.setScene(self.input_image_view_scene)
        self.ui.input_image_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.input_image_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def build_cell_polygons(self):
        # Replace old QGraphicsScene
        self.input_image_view_cell_polygon_items.clear()
        self.input_image_view_scene = ClickableGraphicsScene(self, double_click_callback=self.on_click_input_view)
        self.input_image_view_scene.addItem(self.input_image_pixmap_container)
        self.ui.input_image_view.setScene(self.input_image_view_scene)

        width, height = self.input_image_drawn_size
        for cell_id, cell in self.cell_registry.cells.items():
            contour = cell.contour_normalized
            contour = contour * np.array([width, height])

            polygon = QPolygonF()
            for point in contour:
                polygon << QPointF(*point)
            polygon << QPointF(*contour[0])

            polygon_item = QGraphicsPolygonItem(polygon)
            pen = QPen(QColor(255, 0, 0))
            polygon_item.setPen(pen)
            polygon_item.setVisible(True)

            self.input_image_view_scene.addItem(polygon_item)
            self.input_image_view_cell_polygon_items[cell_id] = polygon_item

    def build_sorting_combo_box(self):
        names = get_all_metric_names()

        def make_sorter(metric_name, descending_order):
            def sorter(cell_reg):
                l = [(cell_reg.get_cell(x).parameters[metric_name], x) for x in cell_reg.get_cells_to_draw()]
                l.sort(key=lambda x: x[0], reverse=descending_order)
                if False:
                    if descending_order:
                        print(f'Sorting by metric \'{metric_name}\' in descending order.')
                    else:
                        print(f'Sorting by metric \'{metric_name}\' in ascending order.')
                return [x[1] for x in l]

            return sorter

        for metric_name in names:
            self.sorter_combo_box_to_sorter_fn.append(make_sorter(metric_name, descending_order=True))
            self.sorter_combo_box_to_sorter_fn.append(make_sorter(metric_name, descending_order=False))
            self.ui.combo_sort_by.addItem(f'{metric_name}, desc.')
            self.ui.combo_sort_by.addItem(f'{metric_name}, asc.')


class FilterDialog(QDialog):
    """ Filter dialog. """

    def __init__(self, parent, cell_registry):
        super().__init__(parent)
        self.main_window = parent
        self.ui = FilterDialogUI()
        self.ui.setupUi(self)

        self._ignore_events = False
        self._num_slider_ticks = 100  # +1 down the line
        self.cell_registry = cell_registry
        self.filters_backup = copy.deepcopy(self.cell_registry.get_filters())

        self.connect_signals_slots()
        self.show()  # Force layouts to compute widget rects

        self.build_parameter_combo_box()

        # self.histogram_scene = None
        # self.histogram_image_pixmap_container = None
        # self.init_histogram_view()

        self.current_filter_index = 0  # Choose first filter as default
        self.parameter_name = ''
        self.value_range = [None, None]
        self.lower_limit = None
        self.upper_limit = None
        self.on_choose_parameter(self.current_filter_index)  # Also gets all the limits and the value range

        self.update()

    def update(self):
        if self.lower_limit is not None:
            self.ui.spin_box_lower_limit.setValue(self.lower_limit)
            self.set_lower_limit_slider_value(self.lower_limit, self.value_range)
        if self.upper_limit is not None:
            self.ui.spin_box_upper_limit.setValue(self.upper_limit)
            self.set_upper_limit_slider_value(self.upper_limit, self.value_range)
        # self.draw_histogram()
        self.update_filter()
        self.main_window.update()

    def accept(self):
        self.update_filter()
        super().accept()

    def reject(self):
        self.cell_registry.set_filters(self.filters_backup)
        super().reject()

    def update_filter(self):
        f = self.cell_registry.get_filter(self.current_filter_index)
        changed = False
        if f.get_lower_bound() != self.lower_limit:
            f.set_lower_bound(self.lower_limit)
            changed |= True
        if f.get_upper_bound() != self.upper_limit:
            f.set_upper_bound(self.upper_limit)
            changed |= True
        if changed:
            self.cell_registry.update_filtering()
            self.main_window.page_manager.reset()

    def get_data(self):
        return self.cell_registry.get_data_for_parameter(self.parameter_name)

    def get_range(self):
        self.value_range = self.cell_registry.get_parameter_range(self.parameter_name)
        if self.value_range[0] is None or self.value_range[1] is None:
            self.set_adjusters_enabled(False)
            return
        else:
            self.set_adjusters_enabled(True)

        # Dilate both by 1%
        length = self.value_range[1] - self.value_range[0]
        self.value_range[0] = max(0.0, self.value_range[0] - 0.01 * length)
        self.value_range[1] = self.value_range[1] + 0.01 * length

    def get_limits(self):
        f = self.cell_registry.get_filter(self.current_filter_index)
        if f.has_lower_bound():
            self.lower_limit = f.get_lower_bound()
        else:
            self.lower_limit = self.value_range[0]
        if f.has_upper_bound():
            self.upper_limit = f.get_upper_bound()
        else:
            self.upper_limit = self.value_range[1]

    def get_lower_limit_slider_value(self, range: [float, float]):
        if range[0] is None or range[1] is None:
            return None
        # print(self.ui.slider_lower_limit.value())
        return range[0] + float(self.ui.slider_lower_limit.value() * (range[1] - range[0])) / self._num_slider_ticks

    def get_upper_limit_slider_value(self, range: [float, float]):
        if range[0] is None or range[1] is None:
            return None
        # print(self.ui.slider_upper_limit.value())
        return range[0] + float(self.ui.slider_upper_limit.value() * (range[1] - range[0])) / self._num_slider_ticks

    def set_adjusters_enabled(self, enabled: bool):
        self.ui.spin_box_lower_limit.setEnabled(enabled)
        self.ui.spin_box_upper_limit.setEnabled(enabled)
        self.ui.slider_lower_limit.setEnabled(enabled)
        self.ui.slider_upper_limit.setEnabled(enabled)

    def set_lower_limit_slider_value(self, value: float, range: [float, float]):
        step = round(float((value - range[0]) * self._num_slider_ticks) / (range[1] - range[0]))
        self.ui.slider_lower_limit.setValue(step)

    def set_upper_limit_slider_value(self, value: float, range: [float, float]):
        step = round(float((value - range[0]) * self._num_slider_ticks) / (range[1] - range[0]))
        self.ui.slider_upper_limit.setValue(step)

    def connect_signals_slots(self):
        # Combo boxes
        self.ui.combo_parameter_name.activated.connect(self.on_choose_parameter)

        # Spin boxes
        self.ui.spin_box_lower_limit.valueChanged.connect(self.on_lower_limit_spin_box)
        self.ui.spin_box_upper_limit.valueChanged.connect(self.on_upper_limit_spin_box)
        self.ui.slider_lower_limit.valueChanged.connect(self.on_lower_limit_slider)
        self.ui.slider_upper_limit.valueChanged.connect(self.on_upper_limit_slider)

    def on_choose_parameter(self, index: int):
        print(f'Change selected filter to filter for parameter {self.cell_registry.get_filter(index).get_name()}.')
        self.current_filter_index = index
        self.parameter_name = self.cell_registry.get_filter(index).get_name()
        self.get_range()
        self.get_limits()

        self._ignore_events = True
        self.ui.spin_box_lower_limit.setRange(*self.value_range)
        self.ui.spin_box_upper_limit.setRange(*self.value_range)
        self.ui.slider_lower_limit.setRange(0, self._num_slider_ticks)
        self.ui.slider_upper_limit.setRange(0, self._num_slider_ticks)
        self._ignore_events = False

        self.update()

    def on_lower_limit_spin_box(self, value: float):
        if self._ignore_events:
            return
        # print(f'Lower limit spin box value changed to {value}.')
        value = min(value, self.value_range[1])
        self.lower_limit = value
        self.upper_limit = max(self.upper_limit, self.lower_limit)
        self.update()

    def on_upper_limit_spin_box(self, value: float):
        if self._ignore_events:
            return
        # print(f'Upper limit spin box value changed to {value}.')
        value = max(value, self.value_range[0])
        self.upper_limit = value
        self.lower_limit = min(self.lower_limit, self.upper_limit)
        self.update()

    def on_lower_limit_slider(self):
        if self._ignore_events:
            return
        value = self.get_lower_limit_slider_value(self.value_range)
        # print(f'Lower limit slider value changed to {value}.')
        if value is None:
            return
        self.lower_limit = value
        self.upper_limit = max(self.upper_limit, self.lower_limit)
        self.update()

    def on_upper_limit_slider(self):
        if self._ignore_events:
            return
        value = self.get_upper_limit_slider_value(self.value_range)
        # print(f'Upper limit slider value changed to {value}.')
        if value is None:
            return
        self.upper_limit = value
        self.lower_limit = min(self.lower_limit, self.upper_limit)
        self.update()

    def build_parameter_combo_box(self):
        entries = [f.get_name() for f in self.cell_registry.get_filters()]
        for entry in entries:
            self.ui.combo_parameter_name.addItem(entry)

    def init_histogram_view(self):
        width, height = get_drawable_rect_size(self.ui.view_histogram)
        pixmap = get_empty_pixmap(width, height, color=0)
        self.histogram_image_pixmap_container = QGraphicsPixmapItem(pixmap)
        self.histogram_scene = ClickableGraphicsScene(self)
        self.histogram_scene.addItem(self.histogram_image_pixmap_container)
        self.ui.view_histogram.setScene(self.histogram_scene)
        self.ui.view_histogram.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.view_histogram.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def draw_histogram(self):
        width, height = get_drawable_rect_size(self.ui.view_histogram)

        DPI = 100
        # print(width, height)
        fig = Figure(figsize=(width / DPI, height / DPI), dpi=DPI)
        ax = fig.add_subplot()
        ax.hist(self.get_data(), color='red', alpha=0.8)
        # plt.show()

        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw', dpi=DPI)
        # print(fig.bbox.bounds)
        plt.close(fig)
        io_buf.seek(0)
        # print(np.frombuffer(io_buf.getvalue(), dtype=np.uint8))
        img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(height, width, 4))
        io_buf.close()
        # print(img_arr.mean(axis=(0, 1)))

        # plt.imshow(img_arr)
        # img_arr = np.zeros_like(img_arr)
        # plt.imshow(img_arr)
        # plt.show()

        image = QImage(img_arr, width, height, width, QImage.Format_RGBA8888)
        pixmap = QPixmap(image)
        self.histogram_image_pixmap_container.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    sys.exit(app.exec())
