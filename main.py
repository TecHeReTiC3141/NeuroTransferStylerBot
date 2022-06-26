from scripts.client_handlers import *


def register_handlers(dp: Dispatcher):
    dp.register_message_handler(start, commands=['start'], state='*')
    dp.register_message_handler(echo_origin, content_types=[ContentType.PHOTO],
                                state=BotStates.origin)
    dp.register_callback_query_handler(load_your_style, text='own_picture',
                                       state=BotStates.origin)
    dp.register_message_handler(style, content_types=[ContentType.PHOTO],
                                state=BotStates.loading_style)


async def on_start(_):
    register_handlers(disp)
    print('Bot is online')
    logging.info('Bot has been launched')


if __name__ == '__main__':
    executor.start_polling(disp, skip_updates=True, on_startup=on_start)
