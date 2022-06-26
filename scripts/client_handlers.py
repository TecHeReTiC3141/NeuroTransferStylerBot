from scripts.bot_init import *
from scripts.bot_states import BotStates

handler_logger = logging.getLogger('client')

async def start(message: Message):
    await message.answer(f'Hello, {message.from_user.username}!')
    await BotStates.origin.set()


async def echo_origin(message: Message):
    photo_ids = set()
    logging.info(message.photo)
    for photo_set in message.photo:
        photo_ids.add(''.join(photo_set.file_id))
        handler_logger.info(photo_set.file_id)

    await bot.send_photo(chat_id=message.from_user.id, photo=photo_ids.pop())
    await BotStates.style.set()

    keyboard = InlineKeyboardMarkup()
    own = InlineKeyboardButton('Your own picture',
                               callback_data='own_picture')
    prepared = InlineKeyboardButton('One of our styles',
                               callback_data='our_styles')
    keyboard.row(own, prepared)
    await message.reply('Please choose how to load style',
                        reply_markup=keyboard)


async def load_your_style(query: CallbackQuery):
    await bot.send_message(query.from_user.id, '')
