import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, CallbackContext, filters
import time
import requests
import datetime

# === IMPORT MACHINE LEARNING ===
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# === CONFIGURAZIONE API FOOTBALL ===
API_FOOTBALL_KEY = "83c7d3ff5945c5b92aa67a698b159e48"
API_URL = "https://v3.football.api-sports.io/fixtures"
HEADERS = {"X-RapidAPI-Key": API_FOOTBALL_KEY}

# Carica i dati dal file CSV
data = pd.read_csv("dati_combinati.csv")

# Filtra per gli ultimi 5 anni
data['MatchDate'] = pd.to_datetime(data['MatchDate'])
recent_data = data[data['MatchDate'] >= pd.Timestamp.now() - pd.DateOffset(years=5)]

# Verifica le colonne richieste
required_columns = [
    'MatchDate', 'HomeTeam', 'FullTimeHomeGoals', 'FullTimeAwayGoals', 'AwayTeam',
    'HomeShots', 'AwayShots', 'HomeCorners', 'AwayCorners',
    'Bet365HomeOdds', 'Bet365AwayOdds', 'Bet365DrawOdds'
]
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    print(f"Le seguenti colonne sono mancanti nel file CSV: {missing_columns}")
    exit()

selected_teams = {}

# === MACHINE LEARNING: ADDDESTRAMENTO MODELLO PER PRONOSTICI ===

# Prepara i dati per il ML (solo partite complete e con dati numerici)
ml_df = recent_data.copy()
ml_df = ml_df.dropna(subset=[
    'FullTimeHomeGoals', 'FullTimeAwayGoals',
    'HomeShots', 'AwayShots',
    'HomeCorners', 'AwayCorners',
    'Bet365HomeOdds', 'Bet365AwayOdds', 'Bet365DrawOdds'
])

def esito_match(row):
    if row['FullTimeHomeGoals'] > row['FullTimeAwayGoals']:
        return 1   # Vittoria casa
    elif row['FullTimeHomeGoals'] < row['FullTimeAwayGoals']:
        return 2   # Vittoria trasferta
    else:
        return 0   # Pareggio

ml_df['esito'] = ml_df.apply(esito_match, axis=1)

ml_features = [
    'HomeShots', 'AwayShots', 'HomeCorners', 'AwayCorners',
    'Bet365HomeOdds', 'Bet365AwayOdds', 'Bet365DrawOdds'
]
X = ml_df[ml_features]
y = ml_df['esito']

# Addestra il modello solo una volta
if len(ml_df) > 50:  # Solo se abbiamo abbastanza dati!
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    ml_model = RandomForestClassifier(n_estimators=80, random_state=42)
    ml_model.fit(X_train, y_train)
else:
    ml_model = None

# === MENU PRINCIPALE ===
async def main_menu(update: Update, context: CallbackContext, new_message=True) -> None:
    text = (
        "🏟️ *Benvenuto nel menu principale!*\n\n"
        "Scegli cosa vuoi fare:"
    )
    keyboard = [
        [InlineKeyboardButton("Analisi Storica Squadre", callback_data="analisi_storica")],
        [InlineKeyboardButton("LIVE 🔴", callback_data="live_matches")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    if new_message:
        await update.message.reply_text(text, reply_markup=reply_markup, parse_mode="Markdown")
    else:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode="Markdown")

# === AVVIO DEL BOT ===
async def start(update: Update, context: CallbackContext) -> None:
    welcome_message = (
        "⚽ **BENVENUTO NEL BOT DI ANALISI SERIE A!** ⚽\n\n"
        "👨‍💻 *Progetto a cura di Luca Cincione come parte della tesi magistrale in Innovazione Digitale e Comunicazione.*\n\n"
        "📈 Questo bot dimostra come i *big data* sportivi possono essere utilizzati per analizzare statistiche, storici e trend delle partite di calcio, "
        "con un focus particolare sulle applicazioni pratiche nell'ambito della comunicazione digitale.\n\n"
        "ℹ️ Puoi:\n"
        "• Consultare lo storico delle partite tra due squadre\n"
        "• Analizzare quote bookmakers\n"
        "• Calcolare pronostici\n"
        "• Visualizzare le partite live di tutto il mondo\n\n"
        "Scegli un'opzione qui sotto per iniziare!"
    )
    keyboard = [
        [InlineKeyboardButton("Analisi Storica Squadre", callback_data="analisi_storica")],
        [InlineKeyboardButton("LIVE 🔴", callback_data="live_matches")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(welcome_message, reply_markup=reply_markup, parse_mode="Markdown")

# === CALLBACK MENU PRINCIPALE ===
async def main_menu_callback(update: Update, context: CallbackContext) -> None:
    await main_menu(update, context, new_message=False)

# === AVVIO ANALISI STORICA (chiede nomi squadre) ===
async def ask_teams(update: Update, context: CallbackContext) -> None:
    text = (
        "✍️ *Inserisci due squadre per analisi storica (es. Milan Inter):*"
        "\n\nScrivi i nomi separati da uno spazio."
    )
    keyboard = [[InlineKeyboardButton("⬅️ Indietro", callback_data="main_menu")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode="Markdown")

# === STORICO PARTITE ===
async def search_matches(update: Update, context: CallbackContext) -> None:
    user_input = update.message.text.strip()

    try:
        home_team, away_team = user_input.split()
        home_team = home_team.lower()
        away_team = away_team.lower()
    except ValueError:
        await update.message.reply_text(
            "❗ *INSERISCI I NOMI DI DUE SQUADRE SEPARATI DA UNO SPAZIO (es. Milan Inter)* ❗",
            parse_mode="Markdown"
        )
        return

    user_id = update.message.from_user.id
    selected_teams[user_id] = (home_team, away_team)

    filtered_data = data[
        ((data["HomeTeam"].str.lower() == home_team) & (data["AwayTeam"].str.lower() == away_team)) |
        ((data["HomeTeam"].str.lower() == away_team) & (data["AwayTeam"].str.lower() == home_team))
    ]

    if not filtered_data.empty:
        response = "🎯 *STORICO DELLE PARTITE TRA LE DUE SQUADRE:*\n\n"
        for _, row in filtered_data.iterrows():
            response += f"📅 {row['MatchDate'].date()}\n"
            response += f"🏟️ {row['HomeTeam']} {row['FullTimeHomeGoals']} - {row['FullTimeAwayGoals']} {row['AwayTeam']}\n\n"

        keyboard = [
            [InlineKeyboardButton("Modalità Avanzata", callback_data="detailed_result")],
            [InlineKeyboardButton("Analisi Quote Bet365", callback_data="bet365_analysis")],
            [InlineKeyboardButton("Pronostico Prossima Partita", callback_data="next_match_prediction")],
            [InlineKeyboardButton("⬅️ Indietro", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            response + "🔄 *Inserisci altre squadre per continuare l'analisi o scegli un'opzione.*",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(
            "❌ *NON HO TROVATO PARTITE STORICHE TRA QUESTE DUE SQUADRE. RIPROVA!* ❌",
            parse_mode="Markdown"
        )

# === MODALITA AVANZATA ===
async def detailed_result(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()

    if user_id not in selected_teams:
        await query.edit_message_text("❗ *Cerca prima lo storico delle partite tra due squadre!* ❗", parse_mode="Markdown")
        return

    home_team, away_team = selected_teams[user_id]

    filtered_data = data[
        ((data["HomeTeam"].str.lower() == home_team) & (data["AwayTeam"].str.lower() == away_team)) |
        ((data["HomeTeam"].str.lower() == away_team) & (data["AwayTeam"].str.lower() == home_team))
    ]

    response = "🎯 *MODALITÀ AVANZATA:*\n\n"
    for _, row in filtered_data.iterrows():
        response += f"📅 {row['MatchDate'].date()}\n"
        response += f"🏟️ {row['HomeTeam']} {row['FullTimeHomeGoals']} - {row['FullTimeAwayGoals']} {row['AwayTeam']}\n"
        response += f"⚽ Tiri: {row['HomeShots']} - {row['AwayShots']}\n"
        response += f"🏐 Corner: {row['HomeCorners']} - {row['AwayCorners']}\n\n"

    keyboard = [
        [InlineKeyboardButton("Analisi Quote Bet365", callback_data="bet365_analysis")],
        [InlineKeyboardButton("Pronostico Prossima Partita", callback_data="next_match_prediction")],
        [InlineKeyboardButton("⬅️ Indietro", callback_data="main_menu")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        response + "🔄 *Inserisci altre squadre o scegli un'opzione.*",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

# === ANALISI QUOTE ===
async def bet365_analysis(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()

    if user_id not in selected_teams:
        await query.edit_message_text("❗ *Cerca prima lo storico delle partite tra due squadre!* ❗", parse_mode="Markdown")
        return

    home_team, away_team = selected_teams[user_id]

    filtered_data = data[
        ((data["HomeTeam"].str.lower() == home_team) & (data["AwayTeam"].str.lower() == away_team)) |
        ((data["HomeTeam"].str.lower() == away_team) & (data["AwayTeam"].str.lower() == home_team))
    ]

    response = "📊 *ANALISI QUOTE BET365:*\n\n"
    for _, row in filtered_data.iterrows():
        home_odds = row['Bet365HomeOdds']
        away_odds = row['Bet365AwayOdds']
        draw_odds = row['Bet365DrawOdds']

        favorite = None
        if home_odds < away_odds and home_odds < draw_odds:
            favorite = row['HomeTeam']
        elif away_odds < home_odds and away_odds < draw_odds:
            favorite = row['AwayTeam']
        else:
            favorite = "Pareggio"

        result = "✔️ Pronostico corretto" if (
            (favorite == row['HomeTeam'] and row['FullTimeHomeGoals'] > row['FullTimeAwayGoals']) or
            (favorite == row['AwayTeam'] and row['FullTimeAwayGoals'] > row['FullTimeHomeGoals']) or
            (favorite == "Pareggio" and row['FullTimeHomeGoals'] == row['FullTimeAwayGoals'])
        ) else "❌ Pronostico errato"

        response += f"📅 {row['MatchDate'].date()}\n"
        response += f"🏟️ {row['HomeTeam']} vs {row['AwayTeam']}\n"
        response += f"📊 Favorita: {favorite}\n"
        response += f"{result}\n\n"

    keyboard = [
        [InlineKeyboardButton("Modalità Avanzata", callback_data="detailed_result")],
        [InlineKeyboardButton("Pronostico Prossima Partita", callback_data="next_match_prediction")],
        [InlineKeyboardButton("⬅️ Indietro", callback_data="main_menu")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        response + "🔄 *Inserisci altre squadre o scegli un'opzione.*",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

# === PRONOSTICO PROSSIMA PARTITA (con ML integrato) ===
async def next_match_prediction(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()

    if user_id not in selected_teams:
        await query.edit_message_text("❗ *Cerca prima lo storico delle partite tra due squadre!* ❗", parse_mode="Markdown")
        return

    home_team, away_team = selected_teams[user_id]
    filtered_data = data[
        ((data["HomeTeam"].str.lower() == home_team) & (data["AwayTeam"].str.lower() == away_team)) |
        ((data["HomeTeam"].str.lower() == away_team) & (data["AwayTeam"].str.lower() == home_team))
    ]

    if not filtered_data.empty:

        # --- MACHINE LEARNING PRONOSTICO ---
        # Ricava i valori medi per le feature ML
        input_data = {}
        for col in ['HomeShots', 'AwayShots', 'HomeCorners', 'AwayCorners', 'Bet365HomeOdds', 'Bet365AwayOdds', 'Bet365DrawOdds']:
            if not filtered_data[col].isnull().all():
                input_data[col] = filtered_data[col].mean()
            else:
                input_data[col] = recent_data[col].mean()

        # Se il modello ML è stato addestrato, usalo!
        if ml_model is not None:
            X_pred = pd.DataFrame([input_data], columns=ml_features)
            proba = ml_model.predict_proba(X_pred)[0]
            # Ordine: [Pareggio, Vittoria Casa, Vittoria Trasferta] se y = 0,1,2
            prob_draw = proba[0] * 100
            prob_home = proba[1] * 100
            prob_away = proba[2] * 100
            ml_msg = (
                "🤖 *Pronostico tramite machine learning:*\n"
                f"🏠 {home_team.capitalize()}: {prob_home:.2f}%\n"
                f"🏃 {away_team.capitalize()}: {prob_away:.2f}%\n"
                f"🤝 Pareggio: {prob_draw:.2f}%\n\n"
            )
        else:
            ml_msg = ""

        # --- PRONOSTICO TRADIZIONALE ---
        total_matches = filtered_data.shape[0]
        home_wins = filtered_data[
            (filtered_data["HomeTeam"].str.lower() == home_team) &
            (filtered_data["FullTimeHomeGoals"] > filtered_data["FullTimeAwayGoals"])
        ].shape[0]
        away_wins = filtered_data[
            (filtered_data["AwayTeam"].str.lower() == away_team) &
            (filtered_data["FullTimeAwayGoals"] > filtered_data["FullTimeHomeGoals"])
        ].shape[0]
        draws = filtered_data[
            filtered_data["FullTimeHomeGoals"] == filtered_data["FullTimeAwayGoals"]
        ].shape[0]

        home_win_rate = (home_wins / total_matches) * 100
        away_win_rate = (away_wins / total_matches) * 100
        draw_rate = (draws / total_matches) * 100

        final_sum = home_win_rate + away_win_rate + draw_rate
        final_home_probability = (home_win_rate / final_sum) * 100
        final_away_probability = (away_win_rate / final_sum) * 100
        final_draw_probability = (draw_rate / final_sum) * 100

        await query.edit_message_text("⏳ *Calcolo del pronostico in corso...*", parse_mode="Markdown")
        time.sleep(1)
        await query.edit_message_text("📊 *Analisi dei dati storici...*", parse_mode="Markdown")
        time.sleep(1)
        await query.edit_message_text("💹 *Incrocio dei dati con le quote...*", parse_mode="Markdown")
        time.sleep(1)

        response = (
            f"{ml_msg}"
            f"📈 *PRONOSTICO STORICO SULLA PROSSIMA PARTITA:*\n\n"
            f"🏠 {home_team.capitalize()}: Probabilità vittoria {final_home_probability:.2f}%\n"
            f"🏃 {away_team.capitalize()}: Probabilità vittoria {final_away_probability:.2f}%\n"
            f"🤝 Pareggio: Probabilità {final_draw_probability:.2f}%\n"
        )
    else:
        response = "❌ *Non ci sono dati sufficienti per fare un pronostico.* ❌"

    keyboard = [
        [InlineKeyboardButton("Modalità Avanzata", callback_data="detailed_result")],
        [InlineKeyboardButton("Analisi Quote Bet365", callback_data="bet365_analysis")],
        [InlineKeyboardButton("⬅️ Indietro", callback_data="main_menu")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        response + "\n🔄 *Scegli un'altra opzione dal menu principale!*",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

# === PARTITE LIVE (TUTTE LE LEGHE) ===
async def show_live_matches(update: Update, context: CallbackContext) -> None:
    if hasattr(update, 'message') and update.message:
        reply = update.message.reply_text
    elif hasattr(update, 'callback_query') and update.callback_query:
        reply = update.callback_query.edit_message_text
        await update.callback_query.answer()
    else:
        return

    params = {"live": "all"}
    try:
        response = requests.get(API_URL, headers=HEADERS, params=params, timeout=10)
        fixtures = response.json().get("response", [])
    except Exception as e:
        await reply("⚠️ Errore nel recupero dei dati live.")
        return

    if not fixtures:
        await reply("⏳ Nessuna partita live trovata al momento.")
        return

    message = "🔴 *PRIME 10 PARTITE LIVE IN TUTTO IL MONDO:*\n\n"
    count = 0
    for match in fixtures:
        if count >= 10:
            break
        league = match.get('league', {})
        league_name = league.get('name', 'Lega Sconosciuta')
        country = league.get('country', 'Paese sconosciuto')
        teams = match['teams']
        fixture = match['fixture']
        status = fixture['status']['short']
        home = teams['home']['name']
        away = teams['away']['name']
        date_str = fixture['date']
        try:
            dt = datetime.datetime.fromisoformat(date_str.replace('Z','+00:00'))
            ora = dt.strftime("%H:%M")
        except Exception:
            ora = "?"
        goals = match['goals']
        score = f"{goals['home']} - {goals['away']}" if (goals['home'] is not None and goals['away'] is not None and status != "NS") else "vs"
        status_map = {
            "NS": "🕒 Inizio ore",
            "1H": "⏱️ 1° Tempo",
            "HT": "⏸️ Intervallo",
            "2H": "⏱️ 2° Tempo",
            "FT": "✅ Finale",
            "ET": "⏱️ Supplementari",
            "P": "⚽ Rigori",
            "LIVE": "🔴 LIVE",
        }
        status_str = status_map.get(status, status)
        message += f"{status_str} {ora} — *{home}* {score} *{away}*\n🏆 {league_name} ({country})\n\n"
        count += 1

    keyboard = [
        [InlineKeyboardButton("⬅️ Indietro", callback_data="main_menu")]
    ]
    await reply(message, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")

# === CALLBACK PER LIVE ===
async def live_matches_callback(update: Update, context: CallbackContext) -> None:
    await show_live_matches(update, context)

def main() -> None:
    TOKEN = "8086882992:AAG-0O8yW3oKv0tifTZNSnNywWznv2ykilI"
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("live", show_live_matches))
    application.add_handler(CommandHandler("menu", main_menu))
    application.add_handler(CallbackQueryHandler(ask_teams, pattern="^analisi_storica$"))
    application.add_handler(CallbackQueryHandler(live_matches_callback, pattern="^live_matches$"))
    application.add_handler(CallbackQueryHandler(main_menu_callback, pattern="^main_menu$"))
    application.add_handler(CallbackQueryHandler(detailed_result, pattern="^detailed_result$"))
    application.add_handler(CallbackQueryHandler(bet365_analysis, pattern="^bet365_analysis$"))
    application.add_handler(CallbackQueryHandler(next_match_prediction, pattern="^next_match_prediction$"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search_matches))

    print("Il bot è in esecuzione...")
    application.run_polling()

if __name__ == "__main__":
    main()
