from bot_init import *
from pprint import pprint

handler_logger = logging.getLogger('client')

async def start(message: Message):

    await message.answer(f'Hello, {message.from_user.username}!')


async def echo_photo(message: Message):
    photo_ids = set()
    logging.info(message.photo)
    for photo_set in message.photo:
        photo_ids.add(''.join(photo_set.file_id))
        handler_logger.info(photo_set.file_id)

    print(photo_ids)
    await bot.send_photo(chat_id=message.from_user.id, photo=photo_ids.pop())

