Surfing through trade-offs
Alex Komoroske
Alex Komoroske

Follow
8 min read
·
Nov 6, 2022
71


2



Let’s say there’s a product team that has shipped a successful product. It’s a bit janky, but customers love it, and they’re constantly adding new features to it with a speed that delights their users.

Quinn recently joined the team from a big company. She’s aghast at how janky the product is, points out falling user satisfaction numbers, and strongly advocates that the team start focusing on quality.

Sam has been on the team since the beginning. He regularly pulls all-nighters to help add features as quickly as possible. He thinks that the reason that the product is successful is because of how quickly they add features. That means it’s imperative that the team to continue to focus on speed, lest they lose sight of what made them successful in the first place.

Who’s right?

When it’s framed as a binary, it’s impossible to tell. Speed is clearly very important, but then again customers really have started complaining about bugs, so quality probably matters, too.


The problem gets significantly easier to resolve when you realize that this is not a binary, but a trade-off. That is, there is a smooth gradient of possibilities between the two extremes. Sometimes you will hear these called dimensions, spectrums, or polarities.


Once you frame it this way it gets easier to reconcile the two different views.

First, where is the team today? It should be pretty easy for everyone to agree that the team is pretty heavily indexed on the speed end.


Where should the team seek to move to? This is where Quinn and Sam disagree.


But when you plot it out, you realize that Quinn and Sam actually aren’t as far off as it seemed! Sam understands some of the value of quality, and Quinn gets that speed to market is one of the key differentiators.

There’s another key thing they both agree on: they both think that the team should move towards the quality end of the spectrum.


Sam thinks the team should move a little, and Quinn thinks the team should move a lot. Luckily, this is easy to resolve, too. The team doesn’t have to make one big decision now; they can slice it up into smaller decisions, making one now and then punting the other decisions until later.


Everyone agrees that they should move towards quality a bit now. Once they get there, they can assess again: do they agree that they should keep on moving towards quality, or do they think they’re good where they are? By then, they’ll have more information (they’ll see how customers respond to their slight tweak in focus) so it will get strictly easier to figure out than it is right now. People intuitively dislike “kicking the can down the road,” but if the road is sloped away from you–that is, the question will get easier to resolve later–then it’s actively good to kick the can!

Finding the right balance point
Where is the best balance point between quality and speed? It depends! There is no context-free correct balance point for any trade-off. It always matters on the specific context you find yourself in: the state of the surrounding system.

The correct balance point is partially defined by what decisions you made in the past, which have in turn likely had some effect on the surrounding system. If the team has been focusing obsessively on speed for a long time, jankiness and bugs will have crept into the product, leading to a stronger pull towards quality as the right balance point. But if the team continually focuses on quality, it’s possible that competitors will start to catch up, meaning that the correct balance point nudges towards speed again.

The balance point will constantly change — sometimes quickly, sometimes slowly, but it will always change. You will never “solve” the balance point, and you must always keep aware of it, at least in the back of your mind, to make sure you notice when it’s time for an adjustment.

The trade-off is not a linear relationship, but rather somewhat of an s-curve. At the extremes an incremental change towards the extreme end will you get a small bit of benefit, but at a large cost in reduction of benefit of the other end. This means that the balance point is almost never at the full extreme end of the trade-off.

Everything is a trade-off
This trick worked well in the case of the disagreement between Sam and Quinn. But surely this trick only works sometimes?

It turns out it works way more often than you think, because everything is a trade-off. Think of something that we typically take as an unalloyed good, like transparency. Surely transparency is always good? But transparency is in tension with candor. If everything is fully transparent then people won’t share candid information with one another — for example, team members won’t share inconvenient information that the boss might not want to hear — and the system will build up stress that might cause a break-down in the future.

If you don’t acknowledge that something is a trade-off, then it is a near guarantee that you will implicitly pick the wrong balance point.

There are many, many different trade-offs that businesses, teams, and individuals are constantly navigating. Here are just a few:


One interesting thing to note: although the trade-offs are all different and somewhat overlapping, they often have a clear orientation that is consistent across the pairs. This means you can pick any one on the left and match it with any one on the right and have a coherent trade-off, even if the fact it’s a trade-off is subtle or not immediately obvious.

Note that different people tend to be drawn to different ends of these trade-offs. For example, TPMs typically are drawn to the “control” side and prototyping engineers are typically drawn to the “autonomy” end. This means that people will have their ego attached to one end of the trade-off, making the debate — especially if it frames the question as a binary and not a trade-off — somewhat emotionally charged.

There are also some situations where there are a triad of inter-related trade-offs, sometimes called an iron triangle. You typically hear these framed as “A, B, or C: pick 2” Examples include the CAP theorem (consistency, availability, and partition tolerance). Another example is the power trade-off in ecosystems between the platform owner, the end user, and the developers. Even in these iron triangle situations, the trade-off is a smooth one, not binary.

Surfing the trade-off
In the example with Sam and Quinn, it made sense to start with a small shift towards quality because that was the perspective on the team with the minimum delta from the current state. But it typically always makes sense to make small, subtle nudges to trade-offs instead of trying to do huge, discontinuous changes.

The reason for this is that the system takes time to react to the changes. You have to wait to see how the overall system reacts before you decide if you should continue moving in a given direction. For example, imagine in the example they had jumped straight to Quinn’s balance point. It could have turned out that their users really do like the speed of new features and don’t mind the jankiness–and then they’d start losing users to their rivals.

The balance point is based on the broader system’s context. That includes what your users out in the wild do, but it also includes what the team itself is doing. It takes time for everyone on a team to “get the message” about a change, especially if the org is large, distributed, or overly busy. In practice this kind of slow diffusion of knowledge can cause a significant lag.

Any time you have a lag time and some amount of uncertainty, you will get oscillation as you constantly overshoot and then undershoot the balance point. If you’ve ever dealt with a shower that takes a particularly long time for the temperature to adjust, you’ll have an intuitive understanding of why this is. This means that the best practice is to make multiple small nudges over time instead of one big shift all at once.


You can think about it like a north-star that you’re sighting off of and surfing towards. The north-star might slide across the sky, dragging the nudges with it, but there’s a layer of indirection that softens any fast changes. Instead of making one big change instantaneously, you smooth the change out over time, which gives you the option value to change course if the conditions change.

Of course, in the real world this rarely happens. Typically a team won’t recognize that there is even a trade-off at all. For example, a team that prioritizes velocity without realizing that they’re accumulating increasing amounts of debt. The pressure builds until it’s obvious to everyone that something drastic must be done, and some leader makes a hard shift to the other end of the spectrum, to “fix the problem once and for all”. That turns out to be too far in the other direction, which means that pressure builds up on that side (e.g. if you’re only paying down debt and not shipping any useful features for users, users will start to complain), necessitating another hard shift in the other direction. This creates a characteristic “infinity swish” as the team jumps back and forth through the extremes continuously.

This infinity swish is inevitable, but the size of it can be made so small as to be almost imperceptible. Think about balancing in a headstand. It isn’t a static balance point, but rather a constant set of intuitive micro-adjustments.

If you do all of this properly, it won’t feel like the system is yanking you back and forth and causing continual chaos or built-up stress; instead it will feel like you’re surfing through the changing context, constantly staying balanced.

