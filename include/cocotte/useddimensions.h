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
    std::list<unsigned int> dimensionsIds;

public:

    // Constructors
    UsedDimensions() = default;
    explicit UsedDimensions(unsigned int totalNbDimensions, std::list<unsigned int> dimensionsIds);
    explicit UsedDimensions(unsigned int totalNbDimensions, std::list<unsigned int> dimensionsIds, unsigned int nbUsed);

    // Main functions
    std::list<unsigned int> const& getIds() const;
    unsigned int getTotalNbDimensions() const;
    unsigned int getNbUsed() const;
    std::list<unsigned int> unusedDimensionsIds() const;
    void addDimension(unsigned int id);

    // Returns a list of combinations of d used dimensions
    std::list<UsedDimensions> getCombinationsFromUsed(unsigned int d) const;
    // Same but with also combinations d-1 used dimensions and an unused one
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
        archive & usedDimensions.nbUsed;
        archive & usedDimensions.totalNbDimensions;
        archive & usedDimensions.dimensionsIds;
    }

    // Display
    friend std::ostream& operator<< (std::ostream& out, UsedDimensions const& uDims)
    {
        out << "[";

        if (uDims.nbUsed > 0)
        {
            auto const dEnd = uDims.dimensionsIds.end();
            auto dIt = uDims.dimensionsIds.begin();

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
