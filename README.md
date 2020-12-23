# How-to-Talk-of-Useful-Distance-Functions

December 22, 2020

I appreciate comments. Shoot me an email at noel_s_cruz@yahoo.com!

Hire me! 😊

- So when we were discussing
nearest neighbor classification,
we kept emphasizing how important it is
to pick the right distance function.
Now it turns out that distance functions
are critical to many different types of machine learning
and that there are a few distance functions
that keep popping up again and again.
So what we'll be doing today is to look
at two families of such distance functions,
the LP norms and metric spaces.
So let's start with LP norms.
What's a good way to measure distance
in M dimensional Euclidean space?
Well, there's L2 distance, which we saw last time.
If you take out a ruler and you measure
the distance between two points, that's L2 distance.
It's a very simple, intuitive distance function
and in many cases, it's the default choice.
But it turns out that L2 is just one member
of a much larger family of distance functions,
called the LP distances.
And here's the general form of an LP distance.
So p can be any number from one to infinity.
And to compute the LP distance between two vectors,
x and z, you'll look at their difference
along each coordinate, you raise it to the p power,
you add this up across all the coordinates,
and then you take the p root of the whole thing.
So, when you plug in p equals two for example,
you just get back the previous formula,
the Euclidean distance.
But when you plug in a different value of p,
like p equals one, for instance,
you get a different distance function.
For p equals one, you get L1 distance,
which is something that is also
used a lot in machine learning.
Okay, so what is this distance function?
Well, let's say we have two points, x and z,
so this is x, and that's z over there.
L2 distance is just distance as the crow flies,
that's the one we've seen.
That's a very simple distance.
In L1 distance, you also wanna get from x to z,
but you're not allowed to go as the crow flies.
You are forced to go along horizontals and verticals.
Okay, so the L1 distance from x to z is this, plus this.
It's the sum of those two legs.
So mathematically, what we do,
is that we look at the difference along each coordinate,
and we simply add up those differences.
So this is the difference along the first coordinate,
this is the difference along the second coordinate,
and we add those two up and that's L1 distance.
A very important distance function.
Another interesting choice is L infinity distance,
which is the last one here in green.
What L infinity distance does,
is to simply look at the single coordinate
along which the distance between x and z is the greatest.
The single coordinate coordinate
with the largest difference xi minus zi.
So in the case of these two points over here,
z and x, the L infinity distance between them,
would simply be this.
So you might wonder how we can plug in p equal to infinity
in that formula over there.
We can't, p has to be a finite number.
But what we can do is to take larger and larger values
of p and then look at the limit.
And that's how we get our L infinity distance.
Okay, so these are three functions to bear in mind.
L1, L2, and L infinity.
And let's see some examples of these now.
Okay, so let's talk about lengths of vectors.
So the length of a vector is simply
its distance from the origin.
So let's say we have a vector over here, which is all ones,
and this is in D dimensional space.
So the vector consists of D1s.
The length of the vector is its distance from the origin,
from the all zeros point.
And we wanna compute its length
in L2, L1, and L infinity.
So let's do L2 first, since that is
the most familiar distance function.
So we wanna compute the L2 norm of x.
Well, plugging into the formula,
the L2 norm is the square root
of the first coordinate squared
plus the second coordinate squared,
all the way to the Dth coordinate squared.
Okay, so the L2 norm of x, is just the square root of D.
Okay, what about the L1 norm of x.
Let's draw this line over here.
The L1 norm of x is the first coordinate absolute value,
plus the second coordinate,
plus all the way to the Dth coordinate.
And so the L1 norm is D.
Now, let's do the L infinity norm.
The L infinity norm of x is the coordinate
along which the value is the largest,
and in this case, one could pick any of the coordinates.
So the L infinity norm is just one.
So the three norms give very different values.
Let's look at another example.
Okay, so now we are in R2, which is the plane,
and we wanna draw all points whose L2 length is one.
So that's something that you might be familiar with,
that's just a circle, okay?
Every point whose length is one.
So in the L2 case, we have a circle.
The unit circle.
Okay, so this point is one, that point is one,
and formally, what we're looking for,
is all points x1, x2, whose L2 norm is one,
in other words, for which x1 squared plus x2 squared
square root is equal to one.
And this is a circle.
This is the formula for the unit circle.
So that's the familiar case.
Now let's try L1.
So we want all points whose length is one.
So we want all points in the plane,
all points x1, x2, and we want the length to be one.
So the absolute value of x1 plus
the absolute value of x2, equals one.
Let's see what points these are.
Okay so let's draw the plane over here.
Okay, so one example of such a point
is the point 1,0.
Because the coordinates add up to one.
Another one is the point 0,1.
Now we're taking absolute value,
so we can also do -1,0 and 0,-1.
So these are four points that lie on this shape.
But we also have, for example, 1/2,1/2,
which lies in the middle over here.
And when we finally join all these points together,
we see that it looks like this.
It has a diamond shape.
Okay, my diamond is a little bit skewed,
but you can imagine what it's supposed to look like.
So that is the unit ball for the L1 norm.
And what is it for L infinity?
Well, this is what it turns out to be.
Instead of a diamond, it just turns out to be a box.
So that's 0, that's 1, 1, -1.
And maybe you can check for yourself
to see that that's really correct.
Okay, so we've talked a little bit about LP norms.
And these are distance functions that really come up a lot,
especially p equals one to an infinity.
So they're gonna turn out to be very useful.
But there are many situations
in which we need other distance functions.
So for example, distance functions that are fine-tuned
to a particular domain.
Or distance functions between objects
that aren't even vectors.
You know, distance functions between strings or graphs.
These sort of things come up all
the time in machine learning.
So is there a broader family of distances,
something that, for example, can be a distance function
on an arbitrary space, not necessarily a space of vectors?
And there are a few such families,
and perhaps the most important of them are distance metrics.
So let's go over the definition of this.
So we have x which is any data space.
It could consist of vectors.
It could be trees.
It could be strings.
Any space, an arbitrary space.
Now we have a distance function on x,
so it's a distance function where you give it
two objects in x and it returns a value.
You know, maybe 3.6 or something like that.
Now, this distance function is called a metric
if it happens to satisfy four properties.
Four basic properties.
The first is that the distances should never be negative.
That sounds fairly reasonable.
The second is, that the distance
from a point to itself should be zero, and moreover,
these are the only cases in which
the distance should be zero.
So it should not be the case that
the distance between two different points is zero.
The third property is that
the distances should be symmetric.
So the distance from x to y should be
the same as the difference from y to x.
And the final property is the triangle inequality.
So what that says is that if you take any three points,
x, y, and z, then the distance from x to y,
is at most the distance, sorry, the distance from x to z
is at most the distance from x to y,
plus the distance from y to z.
And if that holds,
then it satisfies the triangle inequality.
So, any distance function that happens
to satisfy these four properties is called a metric.
And if we find that the distance function
that we are using is a metric, it's useful,
because there are all sorts of things we can do with it.
For example, last time, we talked about
methods for fast nearest neighbor search.
These data structures like ball trees
and k-d trees and locality-sensitive hashing and so on.
Now, many of those methods work only for Euclidean distance.
Work only if you're doing nearest neighbors
in Euclidean space.
But some of them work for arbitrary metrics.
So if the distance function you've chosen
happens to be a metric, then you can use these
to do fast nearest neighbor search.
So, it's very useful to be able
to choose distances that are metrics.
So let's look at some examples
of metric distances.
And the first example is the LP norms.
It turns out that any LP distance is a metric.
So let's pick one concretely.
Let's say L1 distance.
So let's say that the distance between two points
is the L1 distance, and let's say that the points
are in M dimensional space.
So we take the sum over the M coordinates
of the difference between the two vectors
along that coordinate.
That's the L1 distance.
Why is this a metric?
Well, in order to check that,
we just have to go down those four properties
and check them one by one.
So first of all, is this, can this ever be negative?
No, because of the absolute value.
So first property is okay.
If x equals y, is this zero?
Yeah, if x equals y, this is zero.
If this is zero, does that mean x equals y?
Well, if this thing is zero,
it means that all of these absolute values are zero,
which means that x is equal to y.
The second property's fine as well.
Symmetry, is the distance from x to y
the same as the difference from y to x, yes.
And what about the triangle inequality?
Does L1 distance satisfy the triangle inequality?
Well, one thing that's definitely true
is that if you look at any one coordinate,
the distance, the difference between xi and zi
is at most the difference between xi and yi
plus the difference between yi and zi.
And now, we just sum over all coordinates.
And that gives us the triangle inequality.
So L1 distance satisfies all the properties of a metric,
and all the LP distances do.
Okay, now let's look at an example
where the input space does not consist of vectors.
So let's say that we're dealing with strings
over some alphabet.
For instance, if we are dealing with DNA sequences,
then what we have are strings over A, C, G, and T.
That's the input space.
What the star means is strings of arbitrary length
over this alphabet.
So we have two strings.
Let's say A-C-C-G-T
and C-C-G-T.
So we have these two strings.
And now we want a distance function.
What is the distance between x and y?
Now there are many ways in which we can define this,
but from the biologist's point of view,
they will look at these two, and say,
"Well, you know, actually, these are very similar
"because you can get from x to y by just deleting A."
If you were to remove A, they become the same.
Or equivalently, if you were to take y,
and just insert an A at the beginning,
you would get x.
So this kind of distance function
that takes into account insertions, deletions,
and also substitutions, is called edit distance.
And it turns out that it's a metric.
So let's just define it.
The edit distance between two strings, x and y,
is the number of insertions, deletions, and substitutions
needed to get from x to y.
So why is this a metric?
Well, once again, we just have to go through the properties.
So first of all, is it always positive?
Yes.
Is it the case that it's zero if and only if x equals y?
Again, obviously, yes.
Is it symmetric?
Is the number of steps to go from x to y,
the number of insertions, deletions, and substitutions
to go from x to y the same as to go from y to x?
Yes, it is, because a deletion is
the reverse operation of an insertion.
And finally, does it satisfy the triangle inequality?
It does and that's something you can also convince yourself.
So here's an example of a distance function
that's really over a fairly arbitrary space.
A distance function between strings,
but it turns out to be a metric,
which means that we can do all sorts of nice things with it.
So the properties of a metric seem really minimal.
It should be nonnegative.
It should be symmetric.
Should satisfy the triangle inequality.
I mean, would we ever want to use
a distance function that's not a metric?
And, unfortunately, it turns out the answer is yes.
And here, what I've shown here
is the prototypical example of a distance function
that's very widely used and is nothing
close to being a metric.
This is the relative entropy, or the K-L divergence
and it's one of the most standard distance functions
between probability vectors.
So what's a probability vector?
Well, a probability vector is just a vector
that gives probabilities over different outcomes.
So let's say we have four possible outcomes.
An example of a probability vector
is something like 1/2, 1/4, 1/8, 1/8.
It says the probability of outcome one is 1/2.
The probability of outcome two is 1/4.
The probability of outcome three is 1/8.
The probability of outcome four is 1/8.
So it's a bunch of positive numbers that add up to one.
And another probability vector over these four outcomes
might be 1/6, 1/3, 1/3, 1/6.
So how do we measure the distance between
these two probability distributions?
Well, one very natural way of doing that
would be simply to look at the L1 distance
between the vectors p and q.
Or the L2 distance.
And both of those are perfectly reasonable choices.
But, it turns out that in machine learning,
very very often, the distance measure of choice
is the K-L divergence,
which is summarized in this formula over here.
So let's see what that works out to in this case.
So the distance between p and q
is the sum over all coordinates,
so we're now gonna sum over the four coordinates.
The first coordinate, we take 1/2 log 1/2 over 1/6.
So we're looking at the first coordinate over here.
We're comparing the two numbers 1/2 and 1/6.
Now we look at the second coordinate.
We get 1/4, that's p of x, log,
p of x is 1/4, over 1/3
plus 1/8 log 1/8 over 1/3,
plus 1/8 log 1/8 over 1/6.
Very strange.
Now, thankfully, it turns out,
that this can never be negative.
But that's about where the good news ends.
This is not a symmetric distance.
So the distance from p to q is in general
not the same as the difference from q to p.
And it doesn't come close to satisfying
the triangle inequality.
But, it's a distance function we use all the time.
Okay, so what we've done today
is to talk a little bit about distance functions,
and to be honest, we've really just scratched
the surface in terms of organizing
the space of possible distances.
Now complementary to distance functions
are similarity functions.
And that's something that we'll be seeing
a lot more of later on in the course.

I included some posts for reference.

https://github.com/noey2020/How-to-Talk-of-Improving-Nearest-Neighbor

https://github.com/noey2020/How-to-Talk-of-Prediction-Problems

https://github.com/noey2020/How-to-Talk-Matlab-Tricks-and-Tweaks

https://github.com/noey2020/How-to-Talk-Trading-and-Investing

https://github.com/noey2020/How-to-Work-in-Matlab-Development-Environment

https://github.com/noey2020/How-to-Talk-Vaccines

https://github.com/noey2020/How-to-Talk-Regression-in-Matlab

https://github.com/noey2020/How-to-Get-Started-in-Matlab

https://github.com/noey2020/How-to-Convert-Data-from-Web-Service-Using-Matlab

https://github.com/noey2020/Quote-for-the-Day

https://github.com/noey2020/How-to-Talk-Good-Investment-Strategy

https://github.com/noey2020/How-to-Talk-of-Good-Plan

https://github.com/noey2020/Thought-for-the-Day

https://github.com/noey2020/How-to-Talk-Stock-Watch-of-the-Day

https://github.com/noey2020/How-to-Talk-Data-Science

https://github.com/noey2020/How-to-Talk-Fundamental-Analysis

https://github.com/noey2020/How-to-Read-Company-Profiles

https://github.com/noey2020/How-to-Import-Data-from-Spreadsheets-and-Text-Files-Matlab-Without-Coding

https://github.com/noey2020/How-to-Talk-Model-of-Stock-Market-Prices-

https://github.com/noey2020/How-to-Talk-Digital-Wallets

https://github.com/noey2020/How-to-Talk-Investing

https://github.com/noey2020/How-to-Double-Your-Money-in-5years

https://github.com/noey2020/How-to-Talk-Matlab

I appreciate comments. Shoot me an email at noel_s_cruz@yahoo.com!
