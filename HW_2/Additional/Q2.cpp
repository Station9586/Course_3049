#include <bits/stdc++.h>
using namespace std;

// --- Huffman Tree Node Structure ---
struct HuffmanNode {
    char data;            // Character ('\0' for internal nodes)
    int frequency;        // Frequency of the character/subtree
    HuffmanNode *left, *right; // Pointers to children

    // Constructor for leaf nodes
    HuffmanNode(char data, int frequency): data(data), frequency(frequency), left(nullptr), right(nullptr) {}

    // Constructor for internal nodes
    HuffmanNode(HuffmanNode* l, HuffmanNode* r): data('\0'), frequency(l->frequency + r->frequency), left(l), right(r) {}

    // Check if it's a leaf node
    bool isLeaf() const {
        return left == nullptr and right == nullptr;
    }
};

// --- Comparator for Priority Queue (Min Heap) ---
struct cmp {
    bool operator () (const HuffmanNode* n1, const HuffmanNode* n2) {
        // We want the node with LOWER frequency to have HIGHER priority
        return n1->frequency > n2->frequency;
    }
};

// --- Function to calculate character frequencies ---
unordered_map<char, int> calculateFrequencies (const string& text) {
    unordered_map<char, int> freqMap;
    for (char c: text) {
        freqMap[c]++;
    }
    return freqMap;
}

// --- Function to build the Huffman Tree ---
HuffmanNode* buildHuffmanTree (const unordered_map<char, int>& freqMap) {
    // Priority queue to store nodes (min heap based on frequency)
    priority_queue<HuffmanNode*, vector<HuffmanNode*>, cmp> pq;

    // Create a leaf node for each character and add it to the priority queue
    for (const auto &[key, val] : freqMap) {
        if (val > 0) { // Only include characters that actually appear
             pq.push(new HuffmanNode(key, val));
        }
    }

     // Handle edge case: empty text or text with only one unique character
    if (pq.empty()) return nullptr;
    if (pq.size() == 1) {
        // Create a dummy parent node if only one character exists
        HuffmanNode* leaf = pq.top();
        pq.pop();
        // Create an internal node with the leaf as left child (or right, doesn't matter)
        // Assign a dummy character like '\0' or keep the leaf's char if needed, freq is same
        return new HuffmanNode(leaf, new HuffmanNode('\1', 0)); // Use a dummy char like SOH
    }


    // Build the tree by merging nodes
    while (pq.size() > 1) {
        // Extract the two nodes with the lowest frequencies
        HuffmanNode* left = pq.top();
        pq.pop();
        HuffmanNode* right = pq.top();
        pq.pop();

        // Create a new internal node with these two nodes as children
        // The frequency of the new node is the sum of the frequencies of the children
        HuffmanNode* newNode = new HuffmanNode(left, right);

        // Add the new internal node back into the priority queue
        pq.push(newNode);
    }

    // The remaining node is the root of the Huffman Tree
    return pq.top();
}

// --- Function to generate Huffman codes (recursive traversal) ---
void generateCodes (HuffmanNode* root, const string& currentCode, map<char, string>& huffmanCodes) {
    if (!root) {
        return;
    }

    // If it's a leaf node, store the code
    if (root->isLeaf()) {
         // Avoid storing code for the dummy node if created for single char case
        if (root->data != '\1' || root->frequency > 0) {
            huffmanCodes[root->data] = currentCode.empty()? "0" : currentCode; // Handle single node tree case
        }
        return;
    }

    // Traverse left (append '0')
    generateCodes(root->left, currentCode + "0", huffmanCodes);
    // Traverse right (append '1')
    generateCodes(root->right, currentCode + "1", huffmanCodes);
}

// --- Function to encode the input text ---
string encode (const string& text, const map<char, string>& huffmanCodes) {
    string encodedString = "";
    for (char c: text) {
        // Handle case where character might not be in codes (shouldn't happen with proper build)
        if (huffmanCodes.count(c)) {
             encodedString += huffmanCodes.at(c);
        } else {
            cerr << "Warning: Character '" << c << "' not found in Huffman codes map." << '\n';
        }
    }
    return encodedString;
}

// --- Function to decode the encoded text ---
string decode (const string& encodedString, HuffmanNode* root) {
    string decodedString = "";
    if (!root) return ""; // Handle empty tree
    // Handle single node tree case (where root might be the only leaf)
    if (root->isLeaf()) {
        for (size_t i = 0; i < encodedString.length(); ++i) { // Assuming '0'*n encoding
             if (encodedString[i] == '0') {
                 decodedString += root->data;
             } else {
                 cerr << "Warning: Invalid bit in single-node encoded string." << '\n';
             }
        }
        return decodedString;
    }


    HuffmanNode* currentNode = root;
    for (char bit: encodedString) {
        if (bit == '0') {
            currentNode = currentNode->left;
        } else { // bit == '1'
            currentNode = currentNode->right;
        }

        // If reached a leaf node
        if (currentNode && currentNode->isLeaf()) {
            decodedString += currentNode->data;
            currentNode = root; // Go back to the root for the next character
        }
         // Basic error check
        if (!currentNode) {
            cerr << "Error: Invalid encoded sequence or incomplete tree traversal." << '\n';
            break; // Stop decoding
        }
    }
    return decodedString;
}

// --- Function to delete the Huffman Tree (Post-order traversal) ---
void deleteTree (HuffmanNode* node) {
    if (!node) {
        return;
    }
    deleteTree(node->left);
    deleteTree(node->right);
    delete node;
}


int main (void) {
    string text = "Never gonna give you up, never gonna let you down, never gonna run around and desert you.";

    cout << "Original Text: " << text << '\n';
    cout << "Original Size: " << text.length() * 8 << " bits (approx, assuming 8-bit ASCII)" << '\n';

    if (text.empty()) {
        cout << "Input text is empty. Nothing to encode/decode." << '\n';
        return 0;
    }

    // 1. Calculate Frequencies
    unordered_map<char, int> frequencies = calculateFrequencies(text);
    cout << "\nCharacter Frequencies:" << '\n';
    for (const auto &[key, val]: frequencies) {
        cout << "'" << key << "': " << val << '\n';
    }

    // 2. Build Huffman Tree
    HuffmanNode* root = buildHuffmanTree(frequencies);
    if (!root) {
        cerr << "Error building Huffman tree." << '\n';
        return 1;
    }


    // 3. Generate Huffman Codes
    map<char, string> huffmanCodes;
    generateCodes(root, "", huffmanCodes); // Start with empty code string
    cout << "\nHuffman Codes:" << '\n';
    for (const auto &[key, val] : huffmanCodes) {
        cout << "'" << key << "': " << val << '\n';
    }

    // 4. Encode the Text
    string encodedText = encode(text, huffmanCodes);
    cout << "\nEncoded Text: " << encodedText << '\n';
    cout << "Encoded Size: " << encodedText.length() << " bits" << '\n';

    // Calculate compression ratio (simple bit comparison)
    if (!text.empty()) {
        double original_bits = text.length() * 8.0;
        double encoded_bits = encodedText.length();
        if (original_bits > 0) {
            cout << "Compression Ratio (encoded/original bits): " << (encoded_bits / original_bits) << '\n';
        }
    }


    // 5. Decode the Text
    string decodedText = decode(encodedText, root);
    cout << "\nDecoded Text: " << decodedText << '\n';

    // 6. Verify
    if (text == decodedText) {
        cout << "\nVerification Successful: Original and Decoded texts match!" << '\n';
    } else {
        cout << "\nVerification Failed: Texts do not match!" << '\n';
    }

    // 7. Clean up: Delete the Huffman Tree
    deleteTree(root);
    cout << "\nHuffman tree deleted." << '\n';
    return 0;
}