#ifndef _RandomNumGen_hpp
#define _RandomNumGen_hpp


class random_num_gen
{
    private:
        const unsigned int m_a;
        const unsigned long m_max;
        long m_rand;

        inline long next_long_rand(long seed){
            unsigned long low, high;

            low = m_a * (long)(seed & 0xFFFF);
            high = m_a * (long)((unsigned long)seed >> 16);
            low += (high & 0x7FFF) << 16;
            if(low > m_max){
                low &= m_max;
                ++low;
            }
            low += high >> 15;
            if(low > m_max){
                low &= m_max;
                ++low;
            }
            return (long)low;
        }

    public:
        random_num_gen(unsigned long init_seed) :
            m_a(16807),
            m_max(2147483647L),
            m_rand(init_seed ? (init_seed & m_max) : 1)
        {}

        virtual ~random_num_gen() {}

        virtual void seed(unsigned long seed){
            m_rand = (seed ? (seed & m_max) : 1);
        }

        virtual double rand(){
            m_rand = next_long_rand(m_rand);
            return double(m_rand) / double(m_max);
        }
};


#endif