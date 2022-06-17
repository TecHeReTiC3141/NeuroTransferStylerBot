from scripts.bot_init import *
from scripts.client_handlers import *


def register_handlers(dp: Dispatcher):
    dp.register_message_handler(start, commands=['start'])
    dp.register_message_handler(echo_photo, content_types=[ContentType.PHOTO])


async def on_start(_):
    register_handlers(disp)
    print('Bot is online')


if __name__ == '__main__':
    executor.start_polling(disp, skip_updates=True, on_startup=on_start)