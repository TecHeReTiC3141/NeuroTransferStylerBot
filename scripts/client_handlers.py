from scripts.bot_init import *
from scripts.bot_states import *
from scripts.cycleGAN.cyclegan import *
import asyncio


async def start(message: Message):
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True).add('Begin')
    await message.answer(f'''Hello, {message.from_user.username}! 
    I can transfer of one image to another one. Please, try''',reply_markup=keyboard)
    await BotStates.select.set()


async def select_action(message: Message):
    keyboard = InlineKeyboardMarkup()
    style_trans = InlineKeyboardButton('Style Transfering', callback_data='style_transfering')
    h_to_z = InlineKeyboardButton('Turn horse to zebra', callback_data='h2z')
    z_to_h = InlineKeyboardButton('Turn zebra to horse', callback_data='z2h')
    keyboard.row(style_trans, h_to_z, z_to_h)
    await message.reply('What would you like to do?', reply_markup=keyboard)


async def load_your_origin(query: CallbackQuery, state: FSMContext):
    await state.update_data(type=query.data)
    await query.message.reply('Please, load your origin image')
    await BotStates.origin.set()


async def transfer_origin(message: Message, state: FSMContext):
    await bot.send_photo(chat_id=message.from_user.id, photo=message.photo[0].file_id, caption='Your origin')

    origin_file = await bot.get_file(message.photo[0].file_id)
    await state.update_data(origin=origin_file.file_path)
    data = await state.get_data()
    if data['type'] == 'style_transfering':
        await message.reply('Please, load your style image')
        await BotStates.loading_style.set()

    elif data['type'] == 'h2z':
        await message.reply('In progress. Please, wait')

        data = await state.get_data()
        origin_url = bot.get_file_url(data['origin'])

        loop = asyncio.get_event_loop()

        cycleGan = await loop.run_in_executor(None, CycleGAN, 'h2z')
        res = await loop.run_in_executor(None, cycleGan, origin_url)

        keyboard = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True).add('Again')
        await message.answer_photo(photo=res, caption='My output', reply_markup=keyboard)
        await BotStates.select.set()

    elif data['type'] == 'z2h':
        await message.reply('In progress. Please, wait')

        data = await state.get_data()
        origin_url = bot.get_file_url(data['origin'])
        cycleGan = CycleGAN('z2h')
        res = cycleGan(origin_url)

        keyboard = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True).add('Again')
        await message.answer_photo(photo=res, caption='My output', reply_markup=keyboard)
        await BotStates.select.set()


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
    origin, style = data['origin'], data['style']

    origin_url, style_url = bot.get_file_url(origin), bot.get_file_url(style)

    loop = asyncio.get_event_loop()
    transfer = await loop.run_in_executor(None, StyleTransfer, cnn, cnn_normalization_mean, cnn_normalization_std)
    output = await loop.run_in_executor(None, transfer, origin_url, style_url)
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True).add('Again')
    if output == 'Error':
        await message.answer('Problem with loading of image. Please, try again', reply_markup=keyboard)
    else:
        await message.answer_photo(photo=output, caption='My output', reply_markup=keyboard)
    await BotStates.select.set()
