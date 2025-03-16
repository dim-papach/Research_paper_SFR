// Some definitions presupposed by pandoc's typst output.
#let blockquote(body) = [
  #set text( size: 0.92em )
  #block(inset: (left: 1.5em, top: 0.2em, bottom: 0.2em))[#body]
]

#let horizontalrule = [
  #line(start: (25%,0%), end: (75%,0%))
]

#let endnote(num, contents) = [
  #stack(dir: ltr, spacing: 3pt, super[#num], contents)
]

#show terms: it => {
  it.children
    .map(child => [
      #strong[#child.term]
      #block(inset: (left: 1.5em, top: -0.4em))[#child.description]
      ])
    .join()
}

// Some quarto-specific definitions.

#show raw.where(block: true): block.with(
    fill: luma(230), 
    width: 100%, 
    inset: 8pt, 
    radius: 2pt
  )

#let block_with_new_content(old_block, new_content) = {
  let d = (:)
  let fields = old_block.fields()
  fields.remove("body")
  if fields.at("below", default: none) != none {
    // TODO: this is a hack because below is a "synthesized element"
    // according to the experts in the typst discord...
    fields.below = fields.below.amount
  }
  return block.with(..fields)(new_content)
}

#let empty(v) = {
  if type(v) == "string" {
    // two dollar signs here because we're technically inside
    // a Pandoc template :grimace:
    v.matches(regex("^\\s*$")).at(0, default: none) != none
  } else if type(v) == "content" {
    if v.at("text", default: none) != none {
      return empty(v.text)
    }
    for child in v.at("children", default: ()) {
      if not empty(child) {
        return false
      }
    }
    return true
  }

}

#show figure: it => {
  if type(it.kind) != "string" {
    return it
  }
  let kind_match = it.kind.matches(regex("^quarto-callout-(.*)")).at(0, default: none)
  if kind_match == none {
    return it
  }
  let kind = kind_match.captures.at(0, default: "other")
  kind = upper(kind.first()) + kind.slice(1)
  // now we pull apart the callout and reassemble it with the crossref name and counter

  // when we cleanup pandoc's emitted code to avoid spaces this will have to change
  let old_callout = it.body.children.at(1).body.children.at(1)
  let old_title_block = old_callout.body.children.at(0)
  let old_title = old_title_block.body.body.children.at(2)

  // TODO use custom separator if available
  let new_title = if empty(old_title) {
    [#kind #it.counter.display()]
  } else {
    [#kind #it.counter.display(): #old_title]
  }

  let new_title_block = block_with_new_content(
    old_title_block, 
    block_with_new_content(
      old_title_block.body, 
      old_title_block.body.body.children.at(0) +
      old_title_block.body.body.children.at(1) +
      new_title))

  block_with_new_content(old_callout,
    new_title_block +
    old_callout.body.children.at(1))
}

#show ref: it => locate(loc => {
  let target = query(it.target, loc).first()
  if it.at("supplement", default: none) == none {
    it
    return
  }

  let sup = it.supplement.text.matches(regex("^45127368-afa1-446a-820f-fc64c546b2c5%(.*)")).at(0, default: none)
  if sup != none {
    let parent_id = sup.captures.first()
    let parent_figure = query(label(parent_id), loc).first()
    let parent_location = parent_figure.location()

    let counters = numbering(
      parent_figure.at("numbering"), 
      ..parent_figure.at("counter").at(parent_location))
      
    let subcounter = numbering(
      target.at("numbering"),
      ..target.at("counter").at(target.location()))
    
    // NOTE there's a nonbreaking space in the block below
    link(target.location(), [#parent_figure.at("supplement") #counters#subcounter])
  } else {
    it
  }
})

// 2023-10-09: #fa-icon("fa-info") is not working, so we'll eval "#fa-info()" instead
#let callout(body: [], title: "Callout", background_color: rgb("#dddddd"), icon: none, icon_color: black) = {
  block(
    breakable: false, 
    fill: background_color, 
    stroke: (paint: icon_color, thickness: 0.5pt, cap: "round"), 
    width: 100%, 
    radius: 2pt,
    block(
      inset: 1pt,
      width: 100%, 
      below: 0pt, 
      block(
        fill: background_color, 
        width: 100%, 
        inset: 8pt)[#text(icon_color, weight: 900)[#icon] #title]) +
      block(
        inset: 1pt, 
        width: 100%, 
        block(fill: white, width: 100%, inset: 8pt, body)))
}



#let article(
  title: none,
  authors: none,
  date: none,
  abstract: none,
  cols: 1,
  margin: (x: 1.25in, y: 1.25in),
  paper: "us-letter",
  lang: "en",
  region: "US",
  font: (),
  fontsize: 11pt,
  sectionnumbering: none,
  toc: false,
  toc_title: none,
  toc_depth: none,
  doc,
) = {
  set page(
    paper: paper,
    margin: margin,
    numbering: "1",
  )
  set par(justify: true)
  set text(lang: lang,
           region: region,
           font: font,
           size: fontsize)
  set heading(numbering: sectionnumbering)

  if title != none {
    align(center)[#block(inset: 2em)[
      #text(weight: "bold", size: 1.5em)[#title]
    ]]
  }

  if authors != none {
    let count = authors.len()
    let ncols = calc.min(count, 3)
    grid(
      columns: (1fr,) * ncols,
      row-gutter: 1.5em,
      ..authors.map(author =>
          align(center)[
            #author.name \
            #author.affiliation \
            #author.email
          ]
      )
    )
  }

  if date != none {
    align(center)[#block(inset: 1em)[
      #date
    ]]
  }

  if abstract != none {
    block(inset: 2em)[
    #text(weight: "semibold")[Abstract] #h(1em) #abstract
    ]
  }

  if toc {
    let title = if toc_title == none {
      auto
    } else {
      toc_title
    }
    block(above: 0em, below: 2em)[
    #outline(
      title: toc_title,
      depth: toc_depth
    );
    ]
  }

  if cols == 1 {
    doc
  } else {
    columns(cols, doc)
  }
}
#show: doc => article(
  title: [Comparison of Catalogs],
  toc: true,
  toc_title: [Table of contents],
  toc_depth: 3,
  cols: 1,
  doc,
)


= The data
<the-data>
In this script we will compare 2 catalogs, HECATE \(#cite(<kovlakasHeraklionExtragalacticCatalogue2021>);) and LVG \(#cite(<karachentsevUPDATEDNEARBYGALAXY2013>)[, #cite(<karachentsevSTARFORMATIONPROPERTIES2013a>);];)

- The data have been joined based on their position in the sky \(Ra, Dec).
  - We assume that every galaxy within 2 arc seconds of the initial coordinates is the same galaxy.
- We use TOPCAT to create two joins, an inner and an outer join
- We will use the inner join for 1-1 comparisons
- If we see that the data are similar we can use the outer join
- For the comparison we keep the parameters names exactly they are given in the catalogs

The dataset we are going to use for the comparison \(inner join) consists of 288 galaxies and 168 columns with their physical characteristics.

= Catalog Completeness
<catalog-completeness>
Checking for completeness in galaxy catalogs is essential to ensure that the data accurately represents the true population of galaxies, for $z approx 0$. Incomplete catalogs can lead to biased results in statistical studies, such as the distribution of galaxy luminosity, mass, or star formation rates. Additionally, missing galaxies, especially those at faint magnitudes or large distances, can distort cosmological measurements and hinder our understanding of galaxy formation and evolution.

Completeness checks are crucial for addressing selection biases, identify gaps in the data and guide follow-up observations, ensuring that the catalog provides a reliable sample for scientific analysis. Without these checks, conclusions drawn from the data may be inaccurate or incomplete.

== Completeness of the LVG Catalog
<completeness-of-the-lvg-catalog>
The local volume selection has been made by taking into account galaxies with:

+ Radial velocities of

#math.equation(block: true, numbering: "(1)", [ $ V_(L G) < 600 upright(" km") dot.op upright("m")^(- 1) $ ])<eq-lvg-vel>

+ Distances of

#math.equation(block: true, numbering: "(1)", [ $ D < 11 med upright("Mpc") $ ])<eq-lvg-dis>

A simultaneous fulfillment of both conditions \(1) and \(2) is not required.

=== Difficulties in calculating the Completeness
<difficulties-in-calculating-the-completeness>
As explained in \(#cite(<karachentsevUPDATEDNEARBYGALAXY2013>);), it is difficult to calculate the completeness of the catalog.

+ Completeness within a 10 Mpc radius is difficult to assess due to: - Variability in galaxy properties \(luminosity, size, surface brightness, gas content) - Errors in distance measurements \(Tully–Fisher method errors of \~20–25%), especially at the 10 Mpc boundary.
  - Accurate distances are mainly known within \~5 Mpc.
  - Non-Hubble motions \(\~300 km/s)#footnote[#emph[Non-Hubble motion refers to the component of a galaxy’s velocity that deviates from the uniform expansion of the universe, described by the equation] $V = H_0 d + v_p$,where $V$ is the total velocity, $H_0 d$ represents the Hubble flow, and $v_p$​ is the peculiar velocity. #cite(<Essay>);] may make up half of the adopted velocity constraint \(@eq-lvg-vel)
    - Solution for our usage: only keep the galaxies inside a radius of $D = 11 M p c$
  - HI sources in surveys with low angular resolution: "The presence around our Galaxy of hundreds of high-velocity clouds with low line-of-sight velocities and small W50 widths also provokes the inclusion of false “nearby" dwarf galaxies in the LV”\(#cite(<karachentsevUPDATEDNEARBYGALAXY2013>);)

#block[
#set enum(numbering: "1.", start: 2)
+ Astro-Spam and Survey Errors:
  - Automatic surveys produce false detections \(e.g., stars misclassified as galaxies, high-velocity clouds mistaken for dwarfs).
+ Conditional Completeness Estimate: - Galaxies brighter than $M_B^c = - 11^m$ or with linear diameters larger than $A_26 = 1.0$ kpc show \(40–60)% completeness.
  - "among the members of the Local Group \(D \< 1 Mpc), only half of the galaxies have absolute magnitudes brighter than $- 11^m$. Consequently, more than half of the ultra-faint dwarf companions around normal galaxies, like the Sombrero galaxy \(D \= 9.3 Mpc), still remain outside our field of view."
+ Undetected Ultra-Faint Dwarfs:
  - Many faint dwarf galaxies remain undetected beyond the Local Group. Estimated population of undetected dwarfs could be as large as $10^3 dash.en 10^4$ within the LV.
+ Surface Brightness Distribution:
  - Surface brightness remains consistent across distances, except for ultra-faint dwarfs \($S B < 31 upright("mag") dot.op upright("arcsec")^(- 2)$).
  - Faintest dwarf galaxies are detectable only nearby due to their resolution into individual stars.
+ Luminosity-Size Relationship:
  - Observations align with cosmological models predicting $L tilde.op A^3$ \(#cite(<navarroStructureColdDark1996>);), though deviations occur for extremely low surface brightness galaxies.
  - "The deviation from it at the extremely low surface brightness end is due to a systematic overestimation of dwarf galaxy sizes, the brightness profiles of which lie entirely below the Holmberg isophote."
]

== Completeness of HECATE
<completeness-of-hecate>
The completeness of HECATE is difficult to assess due to: - Unknown selection function of HyperLEDA and selection effects from other cross-correlated catalogs. - Estimation based on comparing B-band luminosity distribution with the galaxy luminosity function \(LF).

- HECATE is:

  - Complete down to $L_B tilde.op 10^9.5 L_(B , dot.circle)$ for distances less than 33 Mpc.

  - Complete down to $L_B tilde.op 10^10 L_(B , dot.circle)$ for distances between 67 Mpc and 100 Mpc.

  - Incomplete at distances greater than 167 Mpc, even for the brightest galaxies.

- Completeness estimates based on B-band luminosity density:

  - \>75% complete for distances less than 100 Mpc. $tilde.op 50 %$ complete at distances of $tilde.op 170$ Mpc.

  - Completeness exceeds 100% within 30 Mpc due to the overdensity of galaxies around the Milky Way.

- Completeness in terms of stellar mass \(M\*):

  - Similar to B-band completeness when measured with $K_s$-band luminosity as a tracer for stellar mass.

  - Overdensity at small distances and cut-off at large distances are observed.

- Completeness in terms of star formation rate \(SFR):

  - \~50% complete between 30 and 150 Mpc.

  - Lower SFR completeness due to limitations in WISE-based SFR estimates, which lack full sky coverage.

  - Despite IRAS’s limited depth, it provides \>50% coverage for star-forming galaxies in the local neighborhood.

  - HECATE’s nonuniform SFR and stellar mass coverage, affecting the reliability of stellar population parameters.

In this section we will check the completeness

#figure(
align(center)[#table(
  columns: 2,
  align: (col, row) => (left,right,).at(col),
  inset: 6pt,
  [Table], [Number of galaxies],
  [Inner join],
  [288],
  [Outer join],
  [2901],
  [LVG],
  [1316],
  [HECATE],
  [2901],
  [Unique galaxies in LVG],
  [1028],
  [Unique Galaxies in Hecate],
  [2613],
)]
)

== Completeness of the Inner join
<completeness-of-the-inner-join>
$ upright("Completeness (X)") = upright("(Galaxies in Inner Join)") / upright("(Galaxies in X)") times 100 % $

Completeness \(HECATE)\= 10 %

Completeness \(LVG)\= 22 %

== Completeness in Outer join
<completeness-in-outer-join>
$ upright("Completeness (X)") = upright("(Galaxies in Outer Join form X)") / upright("(Galaxies in X)") times 100 % $

Completeness \(HECATE)\= 90 %

Completeness \(LVG)\= 78 %

Combined Completeness \=$upright("Total galaxies in Outer") / upright("Unique galaxies in HECATE + LVG")$\= 80 %

== Completeness of the Data
<completeness-of-the-data>
#figure([
#block[
#grid(columns: 2, gutter: 2em,
  [
#figure([
#box(width: 366.0pt, image("compare_files/figure-typst/fig-dis-comp-output-1.svg"))
], caption: figure.caption(
position: bottom, 
[
HECATE
]), 
kind: "quarto-subfloat-fig", 
supplement: "", 
numbering: "(a)", 
)
<fig-dis-comp-1>


]
,
  [
#figure([
#box(width: 366.0pt, image("compare_files/figure-typst/fig-dis-comp-output-2.svg"))
], caption: figure.caption(
position: bottom, 
[
LVG
]), 
kind: "quarto-subfloat-fig", 
supplement: "", 
numbering: "(a)", 
)
<fig-dis-comp-2>


]
)
]
], caption: figure.caption(
position: bottom, 
[
Histograms showing the Distance Completeness of the Catalogs
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-dis-comp>


#figure([
#block[
#grid(columns: 2, gutter: 2em,
  [
#figure([
#box(width: 366.0pt, image("compare_files/figure-typst/fig-type-comp-output-1.svg"))
], caption: figure.caption(
position: bottom, 
[
HECATE
]), 
kind: "quarto-subfloat-fig", 
supplement: "", 
numbering: "(a)", 
)
<fig-type-comp-1>


]
,
  [
#figure([
#box(width: 372.0pt, image("compare_files/figure-typst/fig-type-comp-output-2.svg"))
], caption: figure.caption(
position: bottom, 
[
LVG
]), 
kind: "quarto-subfloat-fig", 
supplement: "", 
numbering: "(a)", 
)
<fig-type-comp-2>


]
)
]
], caption: figure.caption(
position: bottom, 
[
Histograms showing the Type Completeness of the Catalogs
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-type-comp>


= How are we going to compare the data?
<how-are-we-going-to-compare-the-data>
For the comparison of our data we will mainly use two methods: linear regression and comparison of the distributions.

For the linear regression out main tools are scatter plots and $R^2$

+ $R^2$: Measures the proportion of variance explained by the linear model.

+ Slope of the Fitted Line: Should be close to 1 for a 1-1 correlation.

+ Pearson Correlation $rho$: Measures the strength and direction of the linear relationship between two variables, ranging from -1 to 1. #footnote[In simple linear regression, $R^2$ is the square of the Pearson correlation coefficient $rho$.]

#block[
#set enum(numbering: "1.", start: 4)
+ Plots: Plots are essential for visually assessing the relationship between two datasets, identifying correlations, trends, and outliers, and evaluating the fit of linear models. Beside scatter plots and histograms, another useful plot is the Correlation Heatmap.

  #quote(block: true)[
  Correlation Heatmaps: A correlation heatmap is a graphical tool that displays the correlation between multiple variables as a color-coded matrix. It’s like a color chart that shows us how closely related different variables are. In a correlation heatmap, each variable is represented by a row and a column, and the cells show the correlation between them. The color of each cell represents the strength and direction of the correlation, with darker colors indicating stronger correlations.
  ]
]

For the distribution comparisons we will use histograms and Kernel Density Estimate \(KDE) plots. The KDE plots visually represent the distribution of data, providing insights into its shape, central tendency, and spread.

Finally, we will examine the percentage change for each galaxy.

- Percentage change: We can calculate the percentage change of the data for each galaxy and then we can see if the data are similar, based on minimum, the maximum and the mean value of the difference.

$ upright("Percentage change") = frac(V_(H e c a t e) - V_(L V G), V_(H e c a t e)) dot.op 100 % $

= Comparable data
<comparable-data>
== Coordinates
<coordinates>
#figure(
align(center)[#table(
  columns: 3,
  align: (col, row) => (center,center,center,).at(col),
  inset: 6pt,
  [LVG], [HECATE], [Description],
  [Dis],
  [D],
  [Distance],
)]
)

#block[
#block[
#figure([
#box(width: 385.5pt, image("compare_files/figure-typst/fig-coord-compare-output-1.svg"))
], caption: figure.caption(
position: bottom, 
[
Comparison of the Distances
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
numbering: "1", 
)
<fig-coord-compare>


]
]
- The average error of the distance in the HECATE catalog is $overline(delta D) = plus.minus 1.6$ Mpc, so the intercept is included in the error.
- So we can assume that the Distances are the same

== Velocities
<velocities>
#figure(
align(center)[#table(
  columns: 3,
  align: (col, row) => (center,center,center,).at(col),
  inset: 6pt,
  [LVG], [HECATE], [Description],
  [RVel \(km/s)],
  [V \(km/s)],
  [Heliocentric radial velocity],
  [VLG \(km/s)],
  [],
  [Radial velocity],
  [cz \(km/s)],
  [],
  [Heliocentric velocity],
  [],
  [V\_VIR \(km/s)],
  [Virgo-infall corrected radial velocity],
)]
)

#block[
#block[
#figure([
#box(width: 371.25pt, image("compare_files/figure-typst/fig-vel-compare-output-1.svg"))
], caption: figure.caption(
position: bottom, 
[
Comparison of the Radial Velocities
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
numbering: "1", 
)
<fig-vel-compare>


]
]
- The average error of the radial velocity in the HECATE catalog is $overline(delta V) = plus.minus 12 upright("km") dot.op s^(- 1)$ , so the intercept is included in the error.
- So we can assume that the radial velocities are the same

#block[
#block[
#figure([
#box(width: 885.75pt, image("compare_files/figure-typst/fig-vel-pairplot-output-1.svg"))
], caption: figure.caption(
position: bottom, 
[
The correlation Matrix of the Velocities. The lower left triangle is composed of the scatter plots of the various velocities, the diagonal shows their distrubution and the upper right triangle shows the correlations of the Velocities
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
numbering: "1", 
)
<fig-vel-pairplot>


]
]
\[?\] The strong correlation between the velocities reported by the HECATE and LVG catalogs \(@fig-vel-pairplot) can be attributed to the fact that both measure the same intrinsic galaxy velocities but reference them to different frames. For instance, velocities can be corrected for the movement of the Earth \(heliocentric frame) or the for the Virgo-infall. These corrections shift the measured velocity slightly, but the fundamental measurement remains consistent. This explains why the velocities are highly correlated, as they represent the same physical phenomenon but adjusted for different reference points.

#pagebreak()
== Morphology and Geometry
<morphology-and-geometry>
#figure(
align(center)[#table(
  columns: 3,
  align: (col, row) => (center,center,center,).at(col),
  inset: 6pt,
  [LVG], [HECATE], [Description],
  [TType],
  [T \(with errors)],
  [Numerical Hubble type following the de Vaucouleurs system],
  [inc],
  [INCL],
  [Inclination \(deg)],
  [a26\_1 \(Major)],
  [R1 \(Semi-major axis)],
  [angular diameter \(arcmin)],
)]
)

=== Galaxy Types
"Morphological type of galaxy in the numerical code according to the classification by de Vaucouleurs et al.~\(1991). It should be noted that about three quarters of objects in the LV are dwarf galaxies, which require a more detailed morphological classification. For example, dwarf spheroidal galaxies and normal ellipticals are usually denoted by the same numerical code T \< 0, although their physical properties drastically differ. The classification problem arises as well for the “transient" type dwarf galaxies, T r, which combine the features of spheroidal \(Sph) and irregular \(Ir)systems. Due to small classification errors, such objects may "jump" from one end of the T scale to the other.” \(#cite(<karachentsevUPDATEDNEARBYGALAXY2013>);)

"as they correspond to types T \< 1. The shortcomings of such a simplified classification of dwarf systems have become apparent; hence, van den Bergh \(1959) proposed a more refined scheme where dwarf galaxies were assumed to be divided by luminosity class…Such an approach allowed us to avoid unpleasant cases in which a dwarf galaxy of intermediate type jumps over due to a misclassification from one end of the de Vaucouleurs sequence \(T \< 1) to the other \(T \= 9, 10)." \(#cite(<karachentsevSTARFORMATIONPROPERTIES2013a>);)

#figure([
#block[
#grid(columns: 2, gutter: 2em,
  [
#figure([
#box(width: 364.5pt, image("compare_files/figure-typst/fig-types-compare-output-1.svg"))
], caption: figure.caption(
position: bottom, 
[
All the galaxies, without the correction
]), 
kind: "quarto-subfloat-fig", 
supplement: "", 
numbering: "(a)", 
)
<fig-types-compare-1>


]
,
  [
#figure([
#box(width: 358.5pt, image("compare_files/figure-typst/fig-types-compare-output-2.svg"))
], caption: figure.caption(
position: bottom, 
[
Comparison with the correction $lr(|T_(H E C A T E) - T_(L V G)|) < sigma_(Tau_(L V G))$
]), 
kind: "quarto-subfloat-fig", 
supplement: "", 
numbering: "(a)", 
)
<fig-types-compare-2>


]
)
]
], caption: figure.caption(
position: bottom, 
[
Comparison of the Types of galaxies
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-types-compare>


- we can assume that the galaxies from the upper left and lower right regions of the plot are classified differently because of this
- this might explain the large errors $delta T$ of HECATE
- why we see a difference in the distribution of \(@fig-type-comp)

We can remove the "problematic" galaxies by only keeping the ones with: $ lr(|T_(H E C A T E) - T_(L V G)|) < sigma_(T_(L V G)) = 4.8 $

- The average uncertainty of the morphological type in the HECATE catalog is $overline(delta T) = plus.minus 1.4$, so the intercept is included in the error.
- So we can assume that the Morphological Types are the same

=== Inclination
#figure([
#block[
#grid(columns: 2, gutter: 2em,
  [
#figure([
#box(width: 356.25pt, image("compare_files/figure-typst/fig-incl-compare-output-1.svg"))
], caption: figure.caption(
position: bottom, 
[
Distribution of the Inclination of the galaxies
]), 
kind: "quarto-subfloat-fig", 
supplement: "", 
numbering: "(a)", 
)
<fig-incl-compare-1>


]
,
  [
#figure([
#box(width: 356.25pt, image("compare_files/figure-typst/fig-incl-compare-output-2.svg"))
], caption: figure.caption(
position: bottom, 
[
Distribution of the Percentage Change
]), 
kind: "quarto-subfloat-fig", 
supplement: "", 
numbering: "(a)", 
)
<fig-incl-compare-2>


]
)
]
], caption: figure.caption(
position: bottom, 
[
Comparison of the Inclination of the galaxies
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-incl-compare>


#block[
#block[
#block[
#figure(
align(center)[#table(
  columns: 4,
  align: (col, row) => (auto,auto,auto,auto,).at(col),
  inset: 6pt,
  [], [inc], [INCL], [Percentage Change \[%\]],
  [count],
  [287],
  [209],
  [202],
  [mean],
  [60],
  [59],
  [-0],
  [std],
  [19],
  [23],
  [0],
  [min],
  [9],
  [0],
  [-3],
  [50%],
  [60],
  [59],
  [0],
  [max],
  [90],
  [90],
  [1],
)]
)

]
]
]
We can see that for values in the range $[tilde.op 30^circle.stroked.tiny , tilde.op 80^circle.stroked.tiny]$, the values of the LVG inclination are higher \(@fig-incl-compare). However, since their means, median, min and maxes are similar and the percentage change is practically 0% \(mean, median, $sigma$ \= 0 with a range $[- 3 % , 1 %]$), we can ignore the differences and assume they are the same values.

=== Major Axis
#figure([
#block[
#grid(columns: 2, gutter: 2em,
  [
#figure([
#box(width: 363.0pt, image("compare_files/figure-typst/fig-axis-compare-output-1.svg"))
], caption: figure.caption(
position: bottom, 
[
Linear scale
]), 
kind: "quarto-subfloat-fig", 
supplement: "", 
numbering: "(a)", 
)
<fig-axis-compare-1>


]
,
  [
#figure([
#box(width: 366.0pt, image("compare_files/figure-typst/fig-axis-compare-output-2.svg"))
], caption: figure.caption(
position: bottom, 
[
$l o g_10$ scale
]), 
kind: "quarto-subfloat-fig", 
supplement: "", 
numbering: "(a)", 
)
<fig-axis-compare-2>


]
)
]
], caption: figure.caption(
position: bottom, 
[
Comparison of the Major Axises of the galaxies
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-axis-compare>


it is not very clear if we truly have a correlation or not \(@fig-axis-compare-1[45127368-afa1-446a-820f-fc64c546b2c5%fig-axis-compare]). We need to see the linear correlation of the decimal logarithms \(@fig-axis-compare-2[45127368-afa1-446a-820f-fc64c546b2c5%fig-axis-compare]).

$overline(R_1) = 3.9$ \[arcmin\], $overline(a_26) = 9.3$ \[arcmin\], so the intercept is negligable.

$ R_1 = 0.48 dot.op a_26 - 0.34 tilde.op 1 / 2 a_26 $

$ log (R 1) = 0.89 log (a_26) - 10^0.38 = log (10^(- 0.38) a_26^0.89) = log (0.41 dot.op a_26^0.89) arrow.r.double R 1 tilde.eq a_26 / 2 $

#pagebreak()
== Luminosities
<luminosities>
#figure(
align(center)[#table(
  columns: 3,
  align: (col, row) => (center,center,center,).at(col),
  inset: 6pt,
  [LVG], [HECATE], [Description],
  [logKLum],
  [logL\_K],
  [],
)]
)

#figure([
#block[
#grid(columns: 2, gutter: 2em,
  [
#figure([
#box(width: 357.0pt, image("compare_files/figure-typst/fig-klum-compare-output-1.svg"))
], caption: figure.caption(
position: bottom, 
[
Linear Regression with free paramaters, $y = a x + b$
]), 
kind: "quarto-subfloat-fig", 
supplement: "", 
numbering: "(a)", 
)
<fig-klum-compare-1>


]
,
  [
#figure([
#box(width: 496.5pt, image("compare_files/figure-typst/fig-klum-compare-output-2.svg"))
], caption: figure.caption(
position: bottom, 
[
Linear Regrassion $y = x$
]), 
kind: "quarto-subfloat-fig", 
supplement: "", 
numbering: "(a)", 
)
<fig-klum-compare-2>


]
)
]
], caption: figure.caption(
position: bottom, 
[
Comparison of the $L_K$ of the galaxies.
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-klum-compare>


$ log (L_(K , H E C)) = 1.16 log (L_(K , L V G)) - 1.78 = log (L_(K , L V G)^1.16 / 10^1.78) arrow.l.r.double L_(K , H E C) = 0.02 dot.op L_(K , L V G)^1.16 $

To assess whether the two galaxy catalogs have a 1-1 correlation, I applied two types of linear fits to the data: a free fit and a forced fit. The free fit allows both the slope and intercept to vary freely, meaning the line can take on any slope and can cross the y-axis at any point. In contrast, for the forced fit, I set the slope to exactly 1 and fixed the line to pass through the origin, meaning the equation becomes $y = x$. This forced fit represents a perfect 1-1 relationship, where the values from one catalog should match the other exactly, without any shift or scaling.

After fitting the data using both methods, I calculated the $R^2$ value for each fit. In this case, both the free fit and the forced fit produced very similar $R^2$ values, around 87.13%#footnote[The difference of the linear regression with the free parameters and the forced regression is $Delta R^2 = 1.14 dot.op 10^(- 13) %$];, indicating that both fits explain about 87% of the variance in the data \(@fig-klum-compare).

The fact that both the free and forced fits have nearly identical $R^2$ values suggests that the data closely follows a 1-1 relationship. In the free fit, the best-fitting line had a very small intercept, which shows that allowing the line to shift vertically did not significantly improve how well the line fits the data. This strongly indicates that the data from the two catalogs are well-aligned and follow the relationship y\=x quite closely, which is what we expect if the catalogs are in good agreement.

While these results are promising, it’s important to note that achieving an $R^2$ of 87%—while high—doesn’t necessarily prove a perfect 1-1 correlation. There may still be small discrepancies or uncertainties in the data that prevent a perfect match. Nonetheless, the fact that fixing the slope to 1 and forcing the line through the origin gave such a good result suggests that the two catalogs are very consistent with each other.

== Magnitudes
<magnitudes>
#figure(
align(center)[#table(
  columns: 3,
  align: (col, row) => (center,center,center,).at(col),
  inset: 6pt,
  [LVG], [HECATE], [Description],
  [mag\_B \(with errors)],
  [BT \(with errors)],
  [],
  [Kmag],
  [K],
  [2MASS band magnitude \(both)],
)]
)

#figure([
#block[
#grid(columns: 2, gutter: 2em,
  [
#figure([
#box(width: 357.0pt, image("compare_files/figure-typst/fig-mag-compare-output-1.svg"))
], caption: figure.caption(
position: bottom, 
[
$M_B$
]), 
kind: "quarto-subfloat-fig", 
supplement: "", 
numbering: "(a)", 
)
<fig-mag-compare-1>


]
,
  [
#figure([
#box(width: 357.75pt, image("compare_files/figure-typst/fig-mag-compare-output-2.svg"))
], caption: figure.caption(
position: bottom, 
[
$M_K$
]), 
kind: "quarto-subfloat-fig", 
supplement: "", 
numbering: "(a)", 
)
<fig-mag-compare-2>


]
)
]
], caption: figure.caption(
position: bottom, 
[
Comparison of the Magnitudes of the galaxies
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-mag-compare>


- $M_B$: it is a 1-1 correlation, since the average error $M_(B , H E C A T E) = 0.4 med m a g$, so the itercept is smaller than the error \(@fig-mag-compare)
- $M_K$: we need to examine it more, since the intercept is bigger than the error $M_(K , H E C A T E) = 0.09 med m a g$ \(@fig-mag-compare)

#figure([
#block[
#grid(columns: 2, gutter: 2em,
  [
#figure([
#box(width: 357.75pt, image("compare_files/figure-typst/fig-kmag-compare-output-1.svg"))
], caption: figure.caption(
position: bottom, 
[
Linear Regression with free paramaters, $y = a x + b$
]), 
kind: "quarto-subfloat-fig", 
supplement: "", 
numbering: "(a)", 
)
<fig-kmag-compare-1>


]
,
  [
#figure([
#box(width: 496.5pt, image("compare_files/figure-typst/fig-kmag-compare-output-2.svg"))
], caption: figure.caption(
position: bottom, 
[
Linear Regrassion $y = x$
]), 
kind: "quarto-subfloat-fig", 
supplement: "", 
numbering: "(a)", 
)
<fig-kmag-compare-2>


]
)
]
], caption: figure.caption(
position: bottom, 
[
Comparison of the $L_K$ of the galaxies.
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-kmag-compare>


Following the same logic as in @fig-klum-compare, we force the parameters of the linear regression to be $y = 1 dot.op x + 0$.

In this case, the data is such that both the forced fit and the free fit explain the variance almost identically \(@fig-kmag-compare-2[45127368-afa1-446a-820f-fc64c546b2c5%fig-kmag-compare]). This can happen when the data roughly follows the pattern $y approx x$, but with a slight bias that the free fit can account for \(in this case, a slope slightly greater than 1 and a negative intercept).

#pagebreak()
== SFR
<sfr>
#figure(
align(center)[#table(
  columns: 3,
  align: (col, row) => (center,center,center,).at(col),
  inset: 6pt,
  [LVG], [HECATE], [Description],
  [],
  [logSFR\_TIR],
  [Decimal logarithm of the total-infrared SFR estimate \[Msol/yr\]],
  [],
  [logSFR\_FIR],
  [Decimal logarithm of the far-infrared SFR estimate \[Msol/yr\]],
  [],
  [logSFR\_60u],
  [Decimal logarithm of the 60um SFR estimate \[Msol/yr\]],
  [],
  [logSFR\_12u],
  [Decimal logarithm of the 12um SFR estimate \[Msol/yr\]],
  [],
  [logSFR\_22u],
  [Decimal logarithm of the 22um SFR estimate \[Msol/yr\]],
  [],
  [logSFR\_HEC],
  [Decimal logarithm of the homogenised SFR estimate \[Msol/yr\]],
  [],
  [logSFR\_GSW],
  [Decimal logarithm of the SFR in GSWLC-2 \[Msol/yr\]],
  [SFRFUV],
  [],
  [FUV derived integral star formation rate],
  [SFRHa],
  [],
  [H{alpha} derived integral star formation rate],
)]
)

#block[
#block[
#figure([
#box(width: 1414.5pt, image("compare_files/figure-typst/fig-sfr-pairplot-output-1.svg"))
], caption: figure.caption(
position: bottom, 
[
Comparison of the $S F R_i$ of the galaxies
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
numbering: "1", 
)
<fig-sfr-pairplot>


]
]
The SFR according to \(#cite(<kroupaConstraintsStarFormation2020>);), can be calculated from the mean of SFR from the Ha and FUV, for $S F R > 10^(- 3) med M_dot.circle thin y r^(- 1)$. As we can see from the plots \(@fig-sfr-lvg) it could be a good aproximation.

#math.equation(block: true, numbering: "(1)", [ $ S F R = frac(S F R_(F U V) + S F R_(H alpha), 2) , upright("if both exist, else: ") S F R = S F R_i , med i = F U V , thin H alpha $ ])<eq-sfr-mean>

#figure([
#block[
#grid(columns: 2, gutter: 2em,
  [
#figure([
#box(width: 356.25pt, image("compare_files/figure-typst/fig-sfr-lvg-output-1.svg"))
], caption: figure.caption(
position: bottom, 
[
Linear scale
]), 
kind: "quarto-subfloat-fig", 
supplement: "", 
numbering: "(a)", 
)
<fig-sfr-lvg-1>


]
,
  [
#figure([
#box(width: 364.5pt, image("compare_files/figure-typst/fig-sfr-lvg-output-2.svg"))
], caption: figure.caption(
position: bottom, 
[
Decimal logarithmic scale
]), 
kind: "quarto-subfloat-fig", 
supplement: "", 
numbering: "(a)", 
)
<fig-sfr-lvg-2>


]
)
]
], caption: figure.caption(
position: bottom, 
[
Comparison of the $S F R_(F U V) - S F R_(H a)$ of the galaxies
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-sfr-lvg>


#figure([
#block[
#grid(columns: 2, gutter: 2em,
  [
#figure([
#box(width: 356.25pt, image("compare_files/figure-typst/fig-sfr-compare-output-1.svg"))
], caption: figure.caption(
position: bottom, 
[
Linear scale
]), 
kind: "quarto-subfloat-fig", 
supplement: "", 
numbering: "(a)", 
)
<fig-sfr-compare-1>


]
,
  [
#figure([
#box(width: 358.5pt, image("compare_files/figure-typst/fig-sfr-compare-output-2.svg"))
], caption: figure.caption(
position: bottom, 
[
Decimal logarithmic scale
]), 
kind: "quarto-subfloat-fig", 
supplement: "", 
numbering: "(a)", 
)
<fig-sfr-compare-2>


]
)
]
], caption: figure.caption(
position: bottom, 
[
Comparison of the SFR’s of the galaxies
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-sfr-compare>


The low correlation in @fig-sfr-compare-1[45127368-afa1-446a-820f-fc64c546b2c5%fig-sfr-compare] and the interval in @fig-sfr-compare-2[45127368-afa1-446a-820f-fc64c546b2c5%fig-sfr-compare] show that the approximation \(@eq-sfr-mean) is not great. According to #cite(<karachentsevSTARFORMATIONPROPERTIES2013a>);:

- #strong[Hα and FUV timescales];: $H_alpha$ traces short-term star formation \(\~10 Myr) via massive O-type stars, while FUV traces longer-term star formation \(\~100 Myr) from B-type stars.

- #strong[Discrepancies due to star formation bursts];: In galaxies with bursty or episodic star formation, Hα can significantly overestimate SFR compared to FUV, which lags in reflecting recent star formation events.

- #strong[Low-mass galaxy variability];: The scatter between $S F R_(upright("H") alpha)$ and $S F R_(upright("FUV"))$ is particularly high in low-mass, dwarf galaxies, where stochastic star formation history causes wide variability between the two indicators.

- #strong[Measurement uncertainties];: Internal extinction, errors in distance estimates, and IMF stochasticity introduce biases in both Hα and FUV fluxes, making simple averaging inappropriate.

- #strong[Conclusion];: The $S F R_(upright("total")) = upright("mean") (S F R_(upright("H") alpha) , S F R_(upright("FUV")))$ method oversimplifies the complexity of star formation processes and is not reliable for galaxies with variable star formation or significant measurement uncertainties

#pagebreak()
== Masses
<masses>
#figure(
align(center)[#table(
  columns: 3,
  align: (col, row) => (center,center,center,).at(col),
  inset: 6pt,
  [LVG], [HECATE], [Description],
  [logM26],
  [],
  [Log mass within Holmberg radius],
  [logMHI],
  [],
  [Log mass within Holmberg radius],
  [],
  [logM\_HEC],
  [Decimal logarithm of the stellar mass \[Msol\]],
  [],
  [logM\_GSW],
  [Decimal logarithm of the stellar mass in GSWLC-2 \[Msol\]],
  [logStellarMass],
  [],
  [Stellar Mass from $M_(\*) \/ L = 0.6$],
)]
)

=== Stellar Masses Comparison
#pagebreak()
#block[
#block[
#figure([
#box(width: 357.0pt, image("compare_files/figure-typst/fig-mass-compare-output-1.svg"))
], caption: figure.caption(
position: bottom, 
[
Comparison of the Stellar Masses of the galaxies
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
numbering: "1", 
)
<fig-mass-compare>


]
]
$ log M_(\* , H E C A T E) = 1.07 dot.op log M_(\* , L V G) - 0.69 , upright(" but ") log M_(\*) > 6 arrow.r.double M_(\* , H E C) tilde.op M_(\* , L V G) $

As we can see the aproximation of Mass/Light\=const.\=0.6 is a pretty good approximation, for the calculation of $log (M_(\*) \/ M_dot.circle)$, especially for high-mass galaxies

#pagebreak()
=== Heatmap
#block[
#block[
#box(width: 709.5pt, image("compare_files/figure-typst/cell-31-output-1.svg"))

]
]
The heatmap is used to visually assess the correlations between different mass estimates of the galaxies, including the mass within the Holmberg radius \($M_26$), the HI gas mass \($M_(H I)$​), the stellar mass from the HECATE catalog \($M_(H E C)$​), and the stellar mass derived assuming a mass-to-light ratio of 0.6 \(StellarMass). The purpose of this visualization is to quickly identify whether these different mass measurements show consistent correlations, which would suggest that they capture similar aspects of the galaxies’ mass distributions.

This quick check using the heatmap helps confirm that, while different catalogs and methods might have their own specific ways of calculating galaxy masses, the overall mass estimates align well, offering a solid basis for further analysis of galaxy properties.

#horizontalrule

#bibliography("../My\_Library.bib")

