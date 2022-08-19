#pragma once
#include <cassert>
#include <vector>

// code adopted from https://stackoverflow.com/q/7185437

class irange {
 public:
   irange(unsigned long end): _begin(0), _end(end), _step(1){}; // single element constructor

   irange(unsigned long begin, unsigned long end, unsigned long step = 1):
    _begin(begin), _end(end), _step(step){
        assert(step != 0 && ((begin > end && step < 0) || step > 0) && "Infinit loop detected");
        assert((end - begin) % step == 0 && "range results in an infinit loop");
    };

    template<class T> 
    irange(const std::vector<T>& v):                            // direct conversion from vector to index iterable range
    _begin(0), _end(v.size()), _step(1){};

   class iterator {
      friend class irange;
    public:
      long int operator *() const { return i_; }
      const iterator &operator ++() { i_ += _step; return *this; }
      iterator operator ++(int) { iterator copy(*this); i_ += _step; return copy; }

      bool operator ==(const iterator &other) const { return i_ == other.i_; }
      bool operator !=(const iterator &other) const { return i_ != other.i_; }

    protected:
      iterator(long int start, long step = 1) : i_ (start), _step(step) { }

    private:
      long i_, _step;
   };

   iterator begin() const { return iterator(_begin, _step); }
   iterator end() const { return iterator(_end); }
  private:
   unsigned long _begin, _end, _step;
};

template <long int T_begin, long int T_end>
class static_irange {
 public:
   class iterator {
      friend class static_range;
    public:
      long int operator *() const { return i_; }
      const iterator &operator ++() { ++i_; return *this; }
      iterator operator ++(int) { iterator copy(*this); ++i_; return copy; }

      bool operator ==(const iterator &other) const { return i_ == other.i_; }
      bool operator !=(const iterator &other) const { return i_ != other.i_; }

    protected:
      iterator(long int start) : i_ (start) { }

    private:
      unsigned long i_;
   };

   iterator begin() const { return iterator(T_begin); }
   iterator end() const { return iterator(T_end); }
};

template <class T>
class enum_range{
  public:
    enum_range(unsigned long start = 0): _begin(start){};
    class iterator{
      public:
        long int operator*() const {return _i;}
        //T operator*() const{return static_cast<T>(_i);}
        const iterator& operator++(){++_i; return *this;}
        iterator operator++(int){iterator copy(*this); ++_i; return copy;}

        bool operator==(const iterator &other) const { return _i == other._i;}
        bool operator!=(const iterator &other) const { return _i != other._i;}

      protected:
        iterator(long int start): _i(start){}
      private:
        unsigned long _i;
    };

    iterator begin() const {return iterator(_begin);}
    iterator end() const {return iterator(static_cast<long>(T::COUNT));}
  private:
    unsigned long _begin;
};