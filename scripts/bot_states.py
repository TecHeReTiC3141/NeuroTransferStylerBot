from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup

class BotStates(StatesGroup):
    origin = State()
    choosing_way_to_load_style = State()
    loading_style = State()
    style = State()
    net_is_working = State()