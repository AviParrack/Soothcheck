Dinosaur Bones and Laser Beams
Approaching complicated and complex problems as constraint satisfactions
Alex Komoroske
Alex Komoroske

Follow
4 min read
·
Jul 13, 2016
10




Problems can have differing proportions of complication and complexity. Figuring out how to make progress in that foggy labyrinth is a key problem for product managers.

Often the key dynamics of a problem can be modeled as a (perhaps ridiculously complex) constraint satisfaction problem. The team is only three engineers and one PM? That’s a constraint. Market research has revealed that the minimum viable product must address a particular use case? That’s a constraint. You have to launch something within three months? You guessed it — that’s a constraint. In practice, a surprising number of complexities of a problem can be modeled as these concrete constraints.

Once all of the concrete constraints are visible, finding the optimal point reduces down to a relatively straightforward constraint satisfaction problem. It may be complicated to solve, but it’s not particularly complex. The exhaustive enumeration of these constraints is the primary difficulty in figuring out a plan.

But real life isn’t always that simple. Although this approach handles complication well, it doesn’t handle complexity. To make it more robust we need to understand two additional types of constraints: 1) concrete but initially hidden, and 2) intangible and permanently hidden. Once we do, we’ll have a way to reduce (at least some) complexity to the easier-to-understand complication.

Dinosaur Bones: Concrete but Initially Hidden Constraints
The first class, concrete but initially hidden, is somewhat straightforward. At the beginning the constraint isn’t obvious, but once you’ve unearthed it, it’s a concrete constraint just like the normal ones. At that point they can be easily added to the set and reasoned about in the same way. These constraints are like dinosaur bones: hidden within the earth, and not at all obvious at the beginning. But once you figure out they’re there and do the painstaking job of digging them out and dusting them off, they’re concrete constraints just like the basic ones were.

Often particularly challenging disagreements come about when you’re standing over these dinosaur bones. Perhaps two team members fundamentally disagree about whether or not a given feature is a good idea. At the beginning agreement may seem impossible. But if you explore the problem via collaborative (as opposed to combative) debate you’ll often unearth the concrete point of disagreement. Perhaps one group thinks that users will skip the text in the onboarding flow, and another group thinks that users will read it. Once you recognize that concrete point of disagreement, you can investigate to figure out the proper course of action. Maybe you run a user study. Maybe you ask other folks who have run into this question before, or do some more research. But once you have a good handle on that question, you’ve successfully dug out the dinosaur bone and the previously complex point of disagreement reduces to a straightforward constraint.

Laser Beams: Intangible and Permanently Hidden Constraints
The next class of constraints is fundamentally challenging to deal with. Dinosaur bones can be vexing when you don’t realize you’re dealing with them, but once you’ve unearthed one it reduces to a run-of-the-mill constraint. The next class of constraints are like laser beams in a security system: impossible to see with the naked eye, and hard to remember even once you’ve discovered them. Unfortunately, these constraints are just as real as straightforward constraints and dinosaur bones.

When you violate a laser beam constraint, you’ll realize something went bad pretty quickly. Perhaps a meeting that’s normally productive devolves into distrustful chaos. Or a straightforward mailing list thread gets escalated and then re-escalated until hundreds of people are slinging passive aggressive jabs at one another.

Even once you’ve violated one of these constraints, it can be hard to realize exactly what went wrong. Seasoned product managers will often have a hunch that a laser beam may be lurking. Conceptually, you can blow smoke in the area where you suspect the laser beam to be, revealing the beam to the naked eye. Once you’ve found it and can demonstrate it to others, they can reason about it almost like a normal constraint.

Perhaps you uncover that two teams from different parts of the company have fundamentally different working styles, with one team preferring moving quickly and adjusting course later, and the other team preferring to take steps only after thinking about them carefully. Perhaps you’ve uncovered that two sub-teams have a different set of incentives that mostly align, but don’t in this particular sub-problem. And of course there’s always the laser beam constraints of ego, pride, or self-interest, which are inescapable aspects of any large team dynamic.

What’s particularly difficult about laser beam constraints is that even once you’ve discovered them, it takes constant vigilance to avoid breaking them since it’s so easy to forget about them. Even if the team has come to understand them, inevitably when a month passes without incident everyone will forget. It is up to the product manager to remain aware of these constraints, and when they see someone inadvertently charging towards one, that they stop them before it’s too late.

Note: The original version of this essay, confusingly, used the concepts of complexity for the easier type, and ambiguity for the harder type, instead of complication vs complexity. I’ve updated the essay to use the word “complexity” in the sense it is used elsewhere, including the cynefin framework, complexity theory, etc.

