#import "@preview/charged-ieee:0.1.3": ieee

// Set global text size for raw text (e.g. code blocks)
#show raw: set text(6pt)

#show: ieee.with(
  title: [A Lockstep Parallel xTB-GFN2 Implementation],
  abstract: [
    Abstract.
  ],
  authors: (
    (
      name: "Anton Marius Nielsen",
      department: [Department of Computer Science],
      organization: [University of Copenhagen],
      //location: [Copenhagen, Denmark],
      //email: "haug@typst.app"
    ),
    (
      name: "Asmus Tørsleff",
      department: [Quantum Information Science],
      organization: [University of Copenhagen],
      //location: [Berlin, Germany],
      //email: "maedje@typst.app"
    ),
  ),
  //index-terms: ("Scientific writing", "Typesetting", "Document creation", "Syntax"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction

== Paper overview

= Methods <sec:methods>
#lorem(45)

$ a + b = gamma $ <eq:gamma>

#lorem(80)

#figure(
  placement: none,
  circle(radius: 15pt),
  caption: [A circle representing the Sun.]
) <fig:sun>

In @fig:sun you can see a common representation of the Sun, which is a star that is located at the center of the solar system.

#lorem(120)

#figure(
  caption: [The Planets of the Solar System and Their Average Distance from the Sun],
  placement: top,
  table(
    // Table styling is not mandated by the IEEE. Feel free to adjust these
    // settings and potentially move them into a set rule.
    columns: (6em, auto),
    align: (left, right),
    inset: (x: 8pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0  { rgb("#efefef") },

    table.header[Planet][Distance (million km)],
    [Mercury], [57.9],
    [Venus], [108.2],
    [Earth], [149.6],
    [Mars], [227.9],
    [Jupiter], [778.6],
    [Saturn], [1,433.5],
    [Uranus], [2,872.5],
    [Neptune], [4,495.1],
  )
) <tab:planets>

In @tab:planets, you see the planets of the solar system and their average distance from the Sun.
The distances were calculated with @eq:gamma that we presented in @sec:methods.

= Parallel Computing

Find out how large nbf can get, aka how many iterations the loops in electro can be. Based on that we would choose between the cpu or gpu version, or only use the cpu version if iterations are at most < 100k or such.

= Lockstep Parallel Electrostatics and Self-Consistent-Charges
```python
def electro(nbf, H0, P, dq, dqsh, atomicGam, shellGam, jmat, shift):

    es = get_isotropic_electrostatic_energy(dq, dqsh, atomicGam, shellGam, jmat, shift)

    k = 0
    h = 0.0
    for i in range(nbf):
        for j in range(i):
            h += P[i,j] * H0[k]
            k += 1
        h += P[i,i] * H0[k] * 0.5
        k += 1

    scc = es + 2.0 * h * evtoau
    return es, scc
```

#v(10pt)

Pseudo code for a lockstep implementation.
```python
def electro(nbf, H0, P, dq, dqsh, atomicGam, shellGam, jmat, shift):
    es = get_isotropic_electrostatic_energy(dq, dqsh, atomicGam, shellGam, jmat, shift)

    h_dev = sycl_malloc(number_of_blocks * block_size)

    ######### KERNEL ##########
    idx = block * blocksize + thread
    
    if (thread < block):
        h_dev[idx] = P[idx] * H0[idx]

    if (thread == block):
        h_dev[idx] = P[idx] * H0[idx] * 0.5

    # barrier
    reduce(h_dev)
    ###########################

    # copy hr from device to host
    h = sycl_move(h_dev)

    scc = es + 2.0 * h * evtoau
    return es, scc
```
