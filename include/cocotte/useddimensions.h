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

public:

    // Constructors
    UsedDimensions() = default; // Necessary for serialization, but do not call it yourself
    explicit UsedDimensions(unsigned int totalNbDimensions,
                            std::list<unsigned int> dimensionsIDs = std::list<unsigned int>{});

    // Main functions
    std::list<unsigned int> const& getIds() const;
    unsigned int getTotalNbDimensions() const;
    unsigned int getNbUsed() const;
    unsigned int getNbUnused() const;

    // Returns a UsedDimensions using every dimensions
    static UsedDimensions allDimensions(unsigned int totalNbDimensions);

    // Returns a UsedDimensions using only the unused dimensions
    UsedDimensions complement() const;

    // Returns a list of combinations of d used dimensions
    std::list<UsedDimensions> getCombinations(unsigned int d) const;

    // Same but with combinations of d0 used dimensions from ud0 and d1 from ud1
    static std::list<UsedDimensions> getMixedCombinations(
            UsedDimensions const& ud0, unsigned int d0, UsedDimensions const& ud1, unsigned int d1);


    // Operators

    // Union
    UsedDimensions const& operator+=(UsedDimensions otherDimensions);

    // Intersection
    UsedDimensions const& operator^=(UsedDimensions otherDimensions);

    friend UsedDimensions operator+(UsedDimensions const& s0, UsedDimensions const& s1)
    {
        UsedDimensions result = s0;
        result += s1;
        return result;
    }

    friend UsedDimensions operator^(UsedDimensions const& s0, UsedDimensions const& s1)
    {
        UsedDimensions result = s0;
        result ^= s1;
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
