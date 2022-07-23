from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup

class BotStates(StatesGroup):
    select = State()
    origin = State()
    cycle_gan_mode = State()
    loading_style = State()
    style = State()
