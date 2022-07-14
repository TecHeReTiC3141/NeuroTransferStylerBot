from aiogram import Bot, Dispatcher, executor
from aiogram.types import *
from aiogram.contrib.fsm_storage.memory import MemoryStorage

bot = Bot('5485009211:AAEacXGZ6_iO60nuIhSw0gaFSt4u_MepgTM')

stor = MemoryStorage()
disp = Dispatcher(bot, storage=stor)
