# ColorBrewer for C++

This header-only library implements the [ColorBrewer](http://colorbrewer2.org/) palette using C++11.

# Usage

Copy `colorbrewer.h` to your project and use it like this:

```cpp
#include "colorbrewer.h"

// To create a std::vector containing strings of hex color codes:
std::vector<const char*> stdColors(brew<const char*>("BuGn", 3));

// To create a QList of QColors: 
QList<QColor> qColors(brew<QColor>("BuGn", 3));
```

# License

Licensed under the [Apache 2.0 License](http://www.apache.org/licenses/LICENSE-2.0).
