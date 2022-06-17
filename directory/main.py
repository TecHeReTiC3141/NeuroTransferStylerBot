from bot_init import *
from client_handlers import *

def register_handlers(dp: Dispatcher):
    dp.register_message_handler(start, commands=['start'])

async def on_start(_):
    register_handlers(disp)
    print('Bot is online')

if __name__ == '__main__':
    executor.start_polling(disp, skip_updates=True, on_startup=on_start)