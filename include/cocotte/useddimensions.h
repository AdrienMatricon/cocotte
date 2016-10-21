#ifndef COCOTTE_USEDDIMENSIONS_H
#define COCOTTE_USEDDIMENSIONS_H


#include <list>
#include <boost/serialization/list.hpp>



namespace Cocotte {



class UsedDimensions
{

private:
    unsigned int nbUsed;
    unsigned int totalNbDimensions;
    std::list<unsigned int> dimensionsIDs;
    int latestAddedDimension = -1;

public:

    // Constructors
    UsedDimensions() = default; // Necessary for serialization, but do not call it yourself
    explicit UsedDimensions(unsigned int totalNbDimensions,
                            std::list<unsigned int> dimensionsIDs = std::list<unsigned int>{});

    // Main functions
    std::list<unsigned int> const& getIds() const;
    unsigned int getTotalNbDimensions() const;
    unsigned int getNbUsed() const;
    std::list<unsigned int> unusedDimensionsIds() const;
    void addDimension(unsigned int id);

    // Accessor and setter
    int getLatestAddedDimension();
    void resetLatestAddedDimension();

    // Returns a UsedDimensions using every dimensions
    static UsedDimensions allDimensions(unsigned int totalNbDimensions);

    // Returns a list of combinations of d used dimensions
    std::list<UsedDimensions> getCombinationsFromUsed(unsigned int d) const;
    // Same but with combinations of d-1 used dimensions and an unused one
    std::list<UsedDimensions> getCombinationsFromUsedAndExactlyOne(unsigned int d) const;
    // Returns both of the above
    std::list<UsedDimensions> getCombinationsFromUsedAndOne(unsigned int d) const;

    // Operators
    UsedDimensions const& operator+=(UsedDimensions otherDimensions);

    friend UsedDimensions operator+(UsedDimensions const& s0, UsedDimensions const& s1)
    {
        UsedDimensions result = s0;
        result += s1;
        return result;
    }

    // Serialization
    template<typename Archive>
    friend void serialize(Archive& archive, UsedDimensions& usedDimensions, const unsigned int version)
    {
        (void) version; // Unused parameter

        archive & usedDimensions.nbUsed;
        archive & usedDimensions.totalNbDimensions;
        archive & usedDimensions.dimensionsIDs;
    }

    // Display
    friend std::ostream& operator<< (std::ostream& out, UsedDimensions const& uDims)
    {
        out << "[";

        if (uDims.nbUsed > 0)
        {
            auto const dEnd = uDims.dimensionsIDs.end();
            auto dIt = uDims.dimensionsIDs.begin();

            out << *dIt;

            while (++dIt != dEnd)
            {
                out << ", " << *dIt;
            }
        }

        out << "] selected from " << uDims.totalNbDimensions << " dimensions";

        return out;
    }
};



}



#endif
