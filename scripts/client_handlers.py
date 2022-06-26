from scripts.bot_init import *
from scripts.bot_states import *

handler_logger = logging.getLogger('client')

async def start(message: Message):
    await message.answer(f'Hello, {message.from_user.username}!')
    await message.answer(f'Please, send original image')
    await BotStates.origin.set()
    print('start')


async def echo_origin(message: Message):
    photo_ids = set()
    logging.info(message.photo)
    for photo_set in message.photo:
        photo_ids.add(''.join(photo_set.file_id))
        handler_logger.info(photo_set.file_id)

    await bot.send_photo(chat_id=message.from_user.id, photo=message.photo[0].file_id)
    # await bot.send_photo(message.from_user.id, message.photo, 'Your original')
    # await state.update_data(origin=message.photo)
    keyboard = InlineKeyboardMarkup()
    own = InlineKeyboardButton('Your own picture',
                               callback_data='own_picture')
    prepared = InlineKeyboardButton('One of our styles',
                               callback_data='our_styles')
    keyboard.row(own, prepared)
    await message.reply('Please choose how to load style',
                        reply_markup=keyboard)
    print(f'loading_origin')


async def load_your_style(query: CallbackQuery):

    await bot.send_message(query.from_user.id, 'Please send picture')
    await BotStates.loading_style.set()
    print(f'loading_style')


async def style(message: Message):

    await bot.send_photo(message.from_user.id, message.photo[0].file_id, 'Your style')
    # await state.update_data(style=message.photo)
    await message.reply('Style transfering is in progress. Please wait')
    await BotStates.net_is_working.set()
    print(f'style')

