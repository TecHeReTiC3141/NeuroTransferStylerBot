from aiogram.types import *

async def start(message: Message):
    
    await message.answer(f'Hello, {message.from_user.username}!')

