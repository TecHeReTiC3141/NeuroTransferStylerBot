from aiogram import Bot, Dispatcher, executor
from aiogram.types import *
from aiogram.contrib.fsm_storage.memory import MemoryStorage


bot = Bot(token='<insert your token here>')

stor = MemoryStorage()
disp = Dispatcher(bot, storage=stor)
