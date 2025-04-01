#include <bits/stdc++.h>
using namespace std;

// --- Global Morse Code Maps ---
map<char, string> text_to_morse_map;
map<string, char> morse_to_text_map;

// Function to Initialize Morse Code Maps
void initializeMorseMaps () {
    // Text to Morse mapping
    text_to_morse_map['A'] = ".-";    text_to_morse_map['B'] = "-...";  text_to_morse_map['C'] = "-.-.";
    text_to_morse_map['D'] = "-..";   text_to_morse_map['E'] = ".";     text_to_morse_map['F'] = "..-.";
    text_to_morse_map['G'] = "--.";   text_to_morse_map['H'] = "....";  text_to_morse_map['I'] = "..";
    text_to_morse_map['J'] = ".---";  text_to_morse_map['K'] = "-.-";   text_to_morse_map['L'] = ".-..";
    text_to_morse_map['M'] = "--";    text_to_morse_map['N'] = "-.";    text_to_morse_map['O'] = "---";
    text_to_morse_map['P'] = ".--.";  text_to_morse_map['Q'] = "--.-";  text_to_morse_map['R'] = ".-.";
    text_to_morse_map['S'] = "...";   text_to_morse_map['T'] = "-";     text_to_morse_map['U'] = "..-";
    text_to_morse_map['V'] = "...-";  text_to_morse_map['W'] = ".--";   text_to_morse_map['X'] = "-..-";
    text_to_morse_map['Y'] = "-.--";  text_to_morse_map['Z'] = "--..";
    text_to_morse_map['0'] = "-----"; text_to_morse_map['1'] = ".----"; text_to_morse_map['2'] = "..---";
    text_to_morse_map['3'] = "...--"; text_to_morse_map['4'] = "....-"; text_to_morse_map['5'] = ".....";
    text_to_morse_map['6'] = "-...."; text_to_morse_map['7'] = "--..."; text_to_morse_map['8'] = "---..";
    text_to_morse_map['9'] = "----.";
    text_to_morse_map['.'] = ".-.-.-"; text_to_morse_map[','] = "--..--"; text_to_morse_map['?'] = "..--..";
    text_to_morse_map['\''] = ".----.";text_to_morse_map['!'] = "-.-.--"; text_to_morse_map['/'] = "-..-.";
    text_to_morse_map['('] = "-.--."; text_to_morse_map[')'] = "-.--.-"; text_to_morse_map['&'] = ".-...";
    text_to_morse_map[':'] = "---..."; text_to_morse_map[';'] = "-.-.-."; text_to_morse_map['='] = "-...-";
    text_to_morse_map['+'] = ".-.-."; text_to_morse_map['-'] = "-....-"; text_to_morse_map['_'] = "..--.-";
    text_to_morse_map['"'] = ".-..-."; text_to_morse_map['$'] = "...-..-";text_to_morse_map['@'] = ".--.-.";

    // Morse to Text mapping (reverse map)
    for (const auto &[key, val]: text_to_morse_map) {
        morse_to_text_map[val] = key;
    }
}

// --- Function to Convert Text to Morse Code ---
string textToMorse (const string& text) {
    stringstream morse_ss;
    bool last_char_was_space = false;

    for (size_t i = 0; i < text.length(); ++i) {
        char c = text[i];

        if (isspace(c)) {
            // If the previous char was not a space, and we have content, add word separator
            if (!last_char_was_space && morse_ss.tellp() > 0) {
                morse_ss << " / "; // Use " / " as word separator
            }
            last_char_was_space = true;
        } else {
            char upper_c = toupper(c);
            if (text_to_morse_map.count(upper_c)) {
                // Add letter separator (single space) before adding the code,
                // unless it's the very beginning or after a word separator.
                if (morse_ss.tellp() > 0 && !last_char_was_space) {
                     morse_ss << " "; // Letter separator
                }
                morse_ss << text_to_morse_map.at(upper_c);
            } else {
                // Handle characters not in the map (optional: print warning or use placeholder)
                 cerr << "Warning: Character '" << c << "' cannot be converted to Morse code. Skipping." << "\n";
            }
            last_char_was_space = false;
        }
    }
    return morse_ss.str();
}

// --- Function to Convert Morse Code to Text ---
string morseToText (const string& morse) {
    stringstream text_ss;
    stringstream morse_stream(morse); // Use stringstream to easily split by spaces
    string current_morse_char;

    // Read Morse code sequences separated by spaces
    while (morse_stream >> current_morse_char) {
        if (current_morse_char == "/") {
            // Handle word separator
            text_ss << " "; // Add a space for the word break
        } else {
            // Look up the Morse sequence in the reverse map
            if (morse_to_text_map.count(current_morse_char)) {
                text_ss << morse_to_text_map.at(current_morse_char);
            } else {
                // Handle unrecognized Morse sequences
                cerr << "Warning: Unrecognized Morse sequence '" << current_morse_char << "'. Using '?'." << "\n";
                text_ss << "?"; // Placeholder for unknown sequences
            }
        }
    }

    // Convert the final text to lowercase (optional, standard practice)
    string result = text_ss.str();
    return result;
}

int main (void) {
    // Initialize the conversion maps
    initializeMorseMaps();

    int choice;
    string input_text;

    cout << "Morse Code Converter" << "\n";
    cout << "--------------------" << "\n";

    while (true) {
        cout << "\nChoose conversion direction:" << "\n";
        cout << "1. Text to Morse Code" << "\n";
        cout << "2. Morse Code to Text" << "\n";
        cout << "3. Exit" << "\n";
        cout << "Enter your choice (1, 2, or 3): ";

        cin >> choice;
        cin.ignore(); // Consume the newline character left by cin

        if (choice == 1) {
            cout << "Enter text to convert to Morse code: ";
            getline(cin, input_text);
            string morse_result = textToMorse(input_text);
            cout << "Morse Code: " << morse_result << "\n";
        } else if (choice == 2) {
            cout << "Enter Morse code to convert to text " << "\n";
            cout << "(Use '.' for dot, '-' for dash, ' ' between letters, ' / ' between words):" << "\n";
            getline(cin, input_text);
            string text_result = morseToText(input_text);
            cout << "Text: " << text_result << "\n";
        } else if (choice == 3) {
            cout << "Exiting program." << "\n";
            break;
        } else {
            cout << "Invalid choice. Please enter 1, 2, or 3." << "\n";
        }
         cout << "--------------------" << "\n";
    }

    return 0;
}