



//              QUERY FOR TESTING


//export const user_query = "Show me the UUIDs of actuator, thermostat and controller";
//export const user_query = "When the temperature exceeds 25°, turn on the air conditioner"
//export const user_query = "Give me a list of the uuid from the 'sensor' devices in the configuration. Indicate the name and category.";
//export const user_query = "Dimmi che dispositivi ci sono nel file";
//export const user_query = "Che sensore mi può dare informazioni di temperatura?";
//export const user_query = "Endpoint per il controllo luci";
/***"Come si chiama il termostato"?
"Che sensore mi può dare informazioni di temperatura"?
"Esistono dei sensori per controllare luci"? */
//export const user_query = "Tell me something about the controller and the firmaware version"; //---> funziona
//export const user_query = "What is the name of thermostat?"
//export const user_query = "Dimmi tutti i parametri del termostato"
//export const user_query = "Give me all parameters of BOX-IO";
//export const user_query = "Show me all the sensors connected to the first floor.";
//export const user_query = "Show me sensors";
//export const user_query = "Show me devices";
//export const user_query = "Accendi le luci";
//export const user_query = "What is the default thermostat setpoint?"; // il "default" rischia di far si che il modello non se la senta di "inventare" perchè non abbiamo un parametro "default" il valore da attribuire
export const user_query = "What is the value of the thermostat setpoint?";
//export const user_query = "Show me all devices located on the second floor";
//export const user_query = "Dimmi qual'è l' UUID del controller luci soggiorno";
//export const user_query = " Shows environmental sensors in the living area that measure both temperature and humidity";

/**A. Test con Query Tipiche:
- "Mostra i sensori di temperatura"
- "UUID del controller luci soggiorno"
- "Parametri configurabili termostato"
- "Dispositivi zona cucina" */


/**
                       QUERY PER AUTOMAZIONE, DA TESTARE

    "Create an automation for the thermostat"

    "When the temperature exceeds 25°, turn on the air conditioner"

    "Schedule the lights to turn on at 6:00 PM"

    "If there is motion, turn on the lights"

*/






/*    1. Query Specifica (Dispositivi)

"Mostrami tutti i termostati presenti nell'impianto con i loro parametri di temperatura"
Verifica:

    Filtraggio preciso per categoria (termostati)

    Estrazione parametri specifici

    Formattazione tabellare

2. Query Generale (Panoramica)

"Descrivi il sistema domotico nel suo complesso"
Verifica:

    Riepilogo non filtrato

    Identificazione categorie principali

    Linguaggio discorsivo

3. Query Comparativa

"Confronta le caratteristiche dei dispositivi BOXIO e WS558"
Verifica:

    Recupero incrociato tra tipologie

    Analisi differenziale

    Tabella comparativa

4. Query Tecnica (Parametri)

"Quali dispositivi misurano il consumo energetico e con quali parametri?"
Verifica:

    Ricerca per parametri specifici

    Mapping dispositivo-parametro

    Precisione tecnica

5. Query Complessa (Multi-filtro)

"Mostra i sensori ambientali nella zona giorno che misurano sia temperatura che umidità"
Verifica:

    Filtri combinati (tipo + posizione + parametri)

    Gestione condizioni complesse

    Rilevanza risultati

Bonus: Query di Fallback

"Cerca dispositivi che non esistono nel sistema" (es. "Mostra le videocamere di sicurezza")
Verifica:

    Gestione errori elegante

    Suggerimenti alternativi

    Riconoscimento limiti del sistema

Ogni query testa un aspetto diverso:

    Precisione (query 1 e 4)

    Sintesi (query 2)

    Analisi (query 3)

    Complessità (query 5)

    Robustezza (bonus)

Per metriche di valutazione, monitora:

    Tempo di risposta

    Numero dispositivi rilevanti restituiti

    Completezza informazioni

    Fluidità linguaggio naturale  */

// export const user_query = "Give me a list of the 'sensor' devices in the configuration. Indicate the name and category.";
