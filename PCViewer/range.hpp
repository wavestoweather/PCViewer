#pragma once
#include <cassert>

// code adopted from https://stackoverflow.com/q/7185437

class irange {
 public:
   irange(unsigned long begin, unsigned long end, unsigned long step = 1):
    _begin(begin), _end(end), _step(step){
        assert(step != 0 && ((begin > end && step < 0) || step > 0) && "Infinit loop detected");
        assert((end - begin) % step == 0 && "range results in an infinit loop");
    };

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