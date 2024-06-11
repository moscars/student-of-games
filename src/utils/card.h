#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <ostream>

struct Card{

    static std::vector<std::string> rankStrings;
    static std::vector<std::string> suitStrings;

    Card() = default;
    Card(int rank_, int suit_) : rank(rank_), suit(suit_) {}
    Card(const std::string & cardString); // Construct a card from a string (e.g. "As" for Ace of Spades)

    int rank;
    int suit;

    bool operator==(const Card& other) const;
    bool operator!=(const Card& other) const;
    bool operator<(const Card& other) const;
    bool operator>(const Card& other) const;
    bool operator<=(const Card& other) const;
    bool operator>=(const Card& other) const;

    int getRank() const;
    int getSuit() const;
    
    std::string toString() const;

};

std::ostream& operator<<(std::ostream& os, const Card& card);