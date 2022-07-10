from scripts.bot_init import *
from scripts.bot_states import *
from scripts.slow_algorithm_of_style_transfering import *

handler_logger = logging.getLogger('client')


async def start(message: Message):
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True).add('Begin')
    await message.answer(f'''Hello, {message.from_user.username}! 
    I can transfer of one image to another one. Please, try''',reply_markup=keyboard)
    await BotStates.select.set()


async def select_action(message: Message):
    keyboard = InlineKeyboardMarkup()
    style_trans = InlineKeyboardButton('Style Transfering', callback_data='style_transfering')
    gan = InlineKeyboardButton('StyleGAN', callback_data='style_gan')
    keyboard.row(style_trans, gan)
    await message.reply('What would you like to do?', reply_markup=keyboard)


async def load_your_origin(query: CallbackQuery):
    await query.message.reply('Please, load your origin image')
    await BotStates.origin.set()


async def transfer_origin(message: Message, state: FSMContext):
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


async def load_your_style(query: CallbackQuery):
    await bot.send_message(query.from_user.id, 'Please send picture')
    await BotStates.loading_style.set()
    print(f'loading_style')


async def style(message: Message, state: FSMContext):
    await bot.send_photo(message.from_user.id, message.photo[0].file_id, 'Your style')
    style = await bot.get_file(message.photo[0].file_id)
    await state.update_data(style=style.file_path)

    await message.reply('Style transfering is in progress. Please wait')

    data = await state.get_data()
    origin, style = data.values()

    origin_url, style_url = bot.get_file_url(origin), bot.get_file_url(style)
    print(origin_url, style_url)

    await message.answer('In progress..')

    transfer = StyleTransfer(cnn, cnn_normalization_mean, cnn_normalization_std)
    output = transfer(origin_url, style_url)

    await message.answer('Result:')
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True).add('Again')
    await message.answer_photo(photo=output, caption='My output', reply_markup=keyboard)
    await BotStates.select.set()
