# Parallel Coordinates Plot (PCP) documentation
Here all settings for the PCP view are explained and demonstrated

### PCP overview
When the PCViewer application is opened, the following can be seen.

Image from standard window

The basic structural elements of the main window are:
1. Menu bar where special settings can be found. All are explained in detail in the [Menu bar section](#menu-bar)



## Menu bar
Image of the menu bar

The menu bar contains the following settings:
1. **Gui**: Gui settings to customize the appearance of the PCViewer application. In the top of the Gui Menu a section for storing and loading styles. With the `Set Default Style` menu item one can select the default style which should be loaded upon starting the PCViewer. In the `Colors` combo selection predefined color settings are safed to give good starting points for customizing the colors. The default style of the PCViewer is the `Dark` color setting.
2. **Settings**: The settings menu tab provides the saved states for PCP settings. Again, similar to the style settings in the Gui tab, one can `Load PCSettings`, `Save/Remove PCSettings` and setting the default PCP settings on starting the application
3. **Maximize**: The maximize menu contains the window width and window height to resize the application to when the `Maximize!` menu item is pressed. This menu was added to provide a way of quickly maximizing the whole application for multiple screens with window managers which do not support multi desktop maximization.,
4. **Attribute**: The Attribute menu manages saved attribute activations and orderings. If the user has found a well suited attribute ordering, activation and attribute axes extremas, it can be saved by either pressing `ctrl+s` or opening this menu and press `Save Attributes`. In a popup modal one can the enter the name for the newly created setting. By default, for each Dataset that is being loaded, a default attribute setting is added, to enable the fast reset of axes scaling, reordering and activation to the standard view on loading. Load a saved Attribute setting got to the `Load` menu item and select the saved settings you want to load. An automatic compatibility check with the current dataset with respect to the attributes is done automatically.
5. **Colors**: The colors menu provides the ability to store colors and load them. To store simply drag and drop a color from anywhere in the application onto the `Colors` menu (Drop indicator will show). Then a popup modal window appears to input a name for the newly saved color. When the color is saved it appears under the `Colors` menu. To load a color simply drag and drop your color out of the list and onto the color field in the application to set the corresponding color.
6. **Global brush**: This menu item contains special settings for global brushing. One can disable global brushing completely by clicking the `Activate Global Brushing`. This also removes the global brush section in the main window. Under `Brush Combination` one can change the combination of global brushes if multiple global brushes are active. With the `Mu add factor` one can set the ratio for minimal brush adjustments. More on that in the [Global brush documentation](brushing.md#global-brushes).
7. **Fractioning**: Here the settings for global brush fractioning can be set. Details on global brush fractioning can be found in the [Glboal brush documentation](brusing.md#global-brush-fractioning). The `Max fraction depth` settings sets the maximum depth for the kd-tree splitting. The `Outlier rank` sets the threshold for split hypercubes in the kd-tree to be seen as outliers and are removed. This functions as a noise reduction technique. The `Bounds behaviour` sets the behaviour of the hyper boxes bounds when a split is done. With this one can set if the fractioned boxes should fit better to the interior points. With the `Split behaviour` the calculation of the cutting plane can be set. `SAH` is the [surface area heuristic](https://pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies#TheSurfaceAreaHeuristic) which tries to keep cluttered datapoints together and separates them from outliers. The `Fractionbox width` sets width of the fraction boxes when being shown on top of the PCP, `Fractionbox linewidth` sets the linewidth of the boxes. The `Multivariate std dev thresh` is the threshold for multivariate brush fractions for which a point is assumed to be inside a brush fraction. For detail on multivariate brush fractions see the [Glboal brush documentation](brusing.md#global-brush-fractioning).
8. **Animation**: In this menu settings for animation are put. On using and capabilities of animations see the [animation section](#animation). The first slider `Animation duration per step` sets the time each step of the animation is displayed. The next check box `Export animation steps` enables the automatic export of the animation steps. For the export the `Export path` has to be set. For the export path the whole path including the filename has to be given. To place the sequence number of the animation step insert `%d` where the number shall be put. The menu item `Start drawlist animation` starts the [drawlist animation](#drawlist-animation). Then  

## Animation
Here Explanations concerning the animation capabilities are given.

Animation here is mainly meant as an automatic run through different visualizations in the PCP. Available animations are:
1. Automatic Drawlist animation, and
2. Automatic Brush animation

### Drawlist Animation

### Brush Animation

## Further read
