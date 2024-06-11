#include "card.h"

std::vector<std::string> Card::rankStrings = {"2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"};
std::vector<std::string> Card::suitStrings = {"c", "d", "h", "s"};

Card::Card(const std::string & cardString){
    rank = std::find(rankStrings.begin(), rankStrings.end(), cardString.substr(0, 1)) - rankStrings.begin();
    suit = std::find(suitStrings.begin(), suitStrings.end(), cardString.substr(1, 1)) - suitStrings.begin();
}

bool Card::operator==(const Card& other) const {
    return rank == other.rank && suit == other.suit;
}
bool Card::operator!=(const Card& other) const {
    return !(*this == other);
}
bool Card::operator<(const Card& other) const {
    return rank < other.rank || (rank == other.rank && suit < other.suit);
}
bool Card::operator>(const Card& other) const {
    return rank > other.rank || (rank == other.rank && suit > other.suit);
}
bool Card::operator<=(const Card& other) const {
    return *this < other || *this == other;
}
bool Card::operator>=(const Card& other) const {
    return *this > other || *this == other;
}

std::string Card::toString() const {
    return rankStrings[rank] + suitStrings[suit];
}

int Card::getRank() const {
    return rank;
}
int Card::getSuit() const {
    return suit;
}

// print
std::ostream& operator<<(std::ostream& os, const Card& card){
    os << card.toString();
    return os;
}
