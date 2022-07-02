from scripts.bot_init import *
from scripts.bot_states import *
from scripts.slow_algorithm_of_style_transfering import *

handler_logger = logging.getLogger('client')

async def start(message: Message):
    await message.answer(f'Hello, {message.from_user.username}!')
    await message.answer(f'Please, send original image')
    await BotStates.origin.set()
    print('start')


async def echo_origin(message: Message, state: FSMContext):
    logging.info(message.photo)

    await bot.send_photo(chat_id=message.from_user.id, photo=message.photo[0].file_id, caption='Your origin')

    origin_file = await bot.get_file(message.photo[0].file_id)
    await state.update_data(origin=origin_file.file_path)
    # await bot.send_photo(message.from_user.id, message.photo, 'Your original')

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


async def style(message: Message, state: FSMContext):

    await bot.send_photo(message.from_user.id, message.photo[0].file_id, 'Your style')
    style = await bot.get_file(message.photo[0].file_id)
    await state.update_data(style=style.file_path)

    await message.reply('Style transfering is in progress. Please wait')
    await BotStates.net_is_working.set()

    data = await state.get_data()
    print(f'1', data)
    origin, style = data.values()
    print(origin, style)
    origin_url, style_url = bot.get_file_url(origin), bot.get_file_url(style)
    print(origin_url, style_url)
    transfer = StyleTransfer(cnn, cnn_normalization_mean, cnn_normalization_std)
    # output = transfer(origin_url, style_url)
