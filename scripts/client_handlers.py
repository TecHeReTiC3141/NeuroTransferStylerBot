from aiogram.types import *
from pprint import pprint

async def start(message: Message):

    await message.answer(f'Hello, {message.from_user.username}!')

async def echo_photo(message: Message):
    pprint(message.photo)
    for photo in message.photo:
        await message.reply(photo)
        print(photo.file_id)

