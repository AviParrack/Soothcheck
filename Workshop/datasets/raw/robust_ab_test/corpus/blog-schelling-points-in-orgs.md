On Schelling Points in Organizations
Tapping into the emergent undercurrents of coordination
Alex Komoroske
Alex Komoroske

Follow
23 min read
·
Dec 13, 2021
257


2



The Coordination Headwind (AKA why organizations are like slime molds, AKA that deck with all of the 🧫🕸🏆) tells the story of how even in the best organizations — with individuals who are good at what they do, hard-working, and collaborative — situations still arise where it’s nearly impossible to get anything done.

The deck only lightly touches on the fact that multiple individuals must pick from among multiple projects at the same time, and that this creates an emergently difficult-to-coordinate situation. However, a deeper understanding of how that dynamic emerges can help navigate real-world organization problems better. A useful way to understand this dynamic is via the lens of schelling points. Once you learn to sense and harness these undercurrents, you can tap into powerful hidden forces.

We’ll first develop an understanding of what this force is by building a toy model that can formally capture the dynamics and then allow us to tinker with it to develop a deeper intuition. We’ll then use the model to dig into tactics that surf this dynamic in real organizations.

Building a toy model
Let’s imagine that there are n different possible projects and m different collaborators. Each collaborator will simultaneously need to pick one of the projects to invest in. A project will only succeed if all of the collaborators pick it. For now let’s assume the collaborators can’t communicate in any way before the decision.

Let’s consider a situation with 4 identical projects and 8 collaborators.


A situation with n projects and m collaborators.
Typical modelling approaches won’t work here because each collaborator will be deciding what to do. Understanding what might happen will require us to simulate those decisions and see how they might affect one another. We’ll simulate 10 runs for each scenario to get a sense for how likely each is to succeed. You can also follow the link in each caption for an interactive version of each scenario that you can experiment with.

In the scenario we’ve described so far, all of the projects are equivalent and there’s no way to communicate, so everyone just picks a random one.


In this situation, the likelihood all collaborators pick the same project is very small. Try the interactive version.
This is a pretty hopeless situation. If there’s a single project (if n is 1), or if there’s a single collaborator (if m is 1) then it’s trivial and will succeed every time. But in any other situation it gets increasingly unlikely that the group will successfully collaborate. In the setup we simulated, they didn’t succeed a single time!

Specifically, the likelihood of success in this model formulation is:


Probability of success is (1/n)^(m-1)
The likelihood of success drops super-linearly as m or n increases. In this toy model, the equilibrium point is failure almost every time for any non-trivial number of projects or collaborators.

So far we’ve assumed that all of the possible projects are indistinguishable (the diagram renders each project in a particular position, but let’s assume that’s not visible to the collaborators). But now let’s imagine that one of the options “stands out” from the others in some notable way. Perhaps it happens to have a scuff mark on it. The mark doesn’t mean anything, per se . It’s just something that makes that one option be unlike the others.


One project has a very small mark on it.
What happens in this case? Before, everyone was equally likely to pick any of the options. But now one of them is obviously different. Not better, per se, just different. Each collaborator is trying to figure out which option the others are likely to pick. Among the field of otherwise-indistinguishable options, there’s now an obvious answer: the one that stands out, the one with the mark on it.


The project with the mark is picked by everyone. Try the interactive version.
This is almost exactly the same situation as the previous hopeless one. The only addition was a random mark that didn’t mean anything. But that was enough to break into a new stable equilibrium in which nearly every trial ends in success. Clearly something very powerful is going on here. The mark formed what is called a schelling point.

Cocktail party trivia… or perhaps more?
The canonical example of a schelling point goes like this. Imagine you and a friend have committed to meet in New York City during the day next Tuesday, but you neglected to coordinate any other specific details. Let’s also imagine that this is taking place in 1960 and no one has cell phones. Where and when do you meet your friend? Noon beneath the clock in Grand Central Terminal. That’s the most distinctive option, the one that the most people are likely to pick.

These schelling points are sometimes created intentionally, and sometimes they emerge organically, like the random scuff mark. They are an emergent phenomena that arises based on everyone trying to reason about what everyone else will choose to do and updating their own choice accordingly. It’s a classic game theoretic scenario.

Of course, in real situations the options are rarely near identical, so this doesn’t seem like a very useful way of looking at the world. The power of the schelling point seems like a theoretical curiosity: a bit of trivia to trot out at cocktail parties, not something that actually applies in real situations in real organizations.

But how do real organizations make decisions about which projects to execute? Upon reflection, the answer seems something like “choosing the best option”. The task of figuring out what project to execute reduces, first and foremost, to figuring out which option is best.

But what does “best” mean? Typically it’s taken to mean the “highest expected bang for buck” —maximizing the expected value divided by expected cost. But what value means is subjective, highly context dependent, and can differ across individuals. For example, a typical organization might ask itself, is value DAUs? Revenue? Attach of another business unit’s product? And value accrued over what time horizon? People might also disagree on how difficult or costly a project might be, changing the “bang for buck” denominator. In addition, there’s the inherent uncertainty of observing facts in a complex and changing environment, let alone predicting things into the future. In practice, coming to wide agreement on which option is “best” in any real world scenario takes an extraordinary amount of effort because the answer is fundamentally nebulous. Even after the fact, reasonable people might disagree about how much value was created.

What if we flipped it on its head? What if instead of thinking about picking an option as first and foremost about figuring out the best option and then coordinating everyone to do it, we instead saw it as first and foremost about figuring out the schelling point that everyone could agree to? With this reframing, we see that agreeing on the best option only happens to be the typical way that groups typically choose what to do. It’s the easy default, because “best” is tied to the ground truth of what will create the most value, and so it’s the strategy individuals will default to. The core dynamic is overcoming the entropy of options to cohere to a schelling point; it just so happens that the most obvious schelling point is trying to discern which is the “best” option, despite “best” being inherently nebulous.

By seeing this coordination first from the angle of a schelling point, and only secondarily about what is “best”, we open ourselves to the possibility that there might be coordination happening around points that are manifestly not the “best” option. Indeed, that happens way more often than we typically think. By opening our eyes to this possibility, we can better predict where coordination challenges will fail, and identify counter-intuitive tactics to increase coordination success even in uncertain environments.

Extending our model
With that motivation, let’s extend the model to be more like real situations so we can use it to discover strategies that we can apply in the real world.

First, let’s allow options to be distinguished by different values. Instead of equal options, each option appears to have some level of goodness or value — some bang for buck. Let’s assume for a minute that everyone agrees on what counts as value in this context and that everyone can all observe the same facts with no uncertainty.


The project that is obviously better is always picked. Try the interactive version.
In this case, even though there are many different options that are roughly the same, there’s still an obvious one that stands out as being best. Everyone picks it, and the equilibrium is success.

However, sometimes there can be multiple good options, with no clear stand out.


There are two very good options that are roughly the same; the collaborators often split between them. Try the interactive version.
The two good options compete for attention and lead to a failure equilibrium.

This is an example of where flipping our perspective to see schelling points as primary helps clarify what’s going on. If you were focused on the best option, having more good options should be better. But having more good options can make things much worse because what matters is how much obviously better one option is: how much it stands out among its peers.

Of course, goodness is not some objective value that everyone sees precisely the same. In real-world environments there is often a large amount of uncertainty about the bang-for-buck of an idea. We can model that as error bars around the value. They can be extremely significant.


Adding error bars makes it hard for even good projects to stand out. Try the interactive version.
By introducing error bars, we’ve made the situation more challenging again. Even if there were a stand-out good project, the collaborators might not agree on it.

So far we’ve been assuming that the collaborators cannot communicate and must simultaneously make their decisions. Let’s extend the model to allow the collaborators to talk amongst themselves, sharing information about their beliefs about each project’s value. At each time step, one person will communicate their beliefs about one project to one other person, who will update their beliefs about that project to some degree. After many such interactions, shared information might diffuse across the network. If there’s sufficient time, it might make it more likely that everyone ultimately picks the same thing. After some number of rounds of communication, everyone will make their choice. In this visualization, the tick marks represent each individual’s belief about the value of a project.


A situation where people can communicate, and over enough time be more likely to pick the same project. Try the interactive version.
Communication can help considerably in some situations, and hurt in others.

Not all interactions are 1:1 or happen with equal likelihood between participants. Let’s add to the model a strength of connection between pairs of collaborators, and expect the stronger pairs to interact more often. Collaborators who happen to be more heavily connected could become focal points. Their beliefs might turn out to be very influential on what the consensus ends up being. You could take this even farther and assume that some collaborators are way more visible than others; the team lead, speaking at an all hands, for example, will immediately influence a large number of people all at once.


A situation where some collaborators can broadcast more information to people simultaneously. Try the interactive version.
All of this communication and coordination takes time, and during that time, the conditions might change. Perhaps the value is tied to what other teams, or even competitors, are doing. Perhaps someone new joins the team, or there’s a reorg and the senior lead is swapped with someone who has a different conception of what the strategy should be. Perhaps a strong-willed collaborator shares their opinion behind the scenes that the official strategy is flawed because it is based on incorrect information. All of these bring ambiguity back into the decision.

The faster the context changes, and the less there’s an unambiguous stand out best option, the more time will be spent frantically trying to coordinate. This coordination is excruciating because it doesn’t feel like making progress, kind of like being stuck in traffic. The longer it drags on, the more everyone gets irritable and frustrated. That frustration causes everyone to frantically try to pick something — anything — to move forward, which creates more thrash, which makes things more uncertainty, in an accelerating spiral of churn. Fighting this entropy requires significant investments of energy. This is the dreaded job of “herding cats”.

There are many more complications we could add to the model — for example, we could introduce the notion of repeated decisions. However, we’ll leave it there for now because we’ve developed a model with enough moving parts to concretely reason about a number of patterns that can help tame this dynamic.

A tour of patterns for real organizations
Pattern: Reduce the number of collaborators

Fewer collaborators makes success more likely. Try the interactive version.
The more people that have to collaborate on the decision, the more likely that the group fails to select the same option. That means an obvious approach is to reduce the number of collaborators.

Collaborators, here, doesn’t mean “people who must execute on the decision” but rather “people who have some role to play in the decision.” In a strongly bottom-up culture, effectively everyone in the team will collaborate on the decision. The other extreme is to have a single decider who makes the call that everyone else will follow. This allows decision-making clarity, but means the whole team is extremely dependent on the quality of that one person’s decisions. Most decisions in complex environments require multiple perspectives to make the right decision because no one person has enough information. Speed and clarity of decision (from having a small number of deciders) is thus in tension with quality and resilience of decision (from having a more diverse set of deciders). A common optimal balance point is captured in the “two pizza team” heuristic: between 6 and 10 people.

Pattern: Reduce the number of options

Fewer options makes it more likely to succeed. Try the interactive version.
The likelihood of failure goes up the more options there are for the collaborators to choose from, so another obvious approach is to reduce the number of options being considered.

The wider the horizon of possible options, the harder it is to coordinate on a strategy, because there are many good (or at least, not obviously bad) options to choose from. Reducing the field of possibility can help focus the discussion. It will feel like trading off optionality (what if one of the curtailed options would have been a breakout success?) but it will allow the team to move more quickly and execute instead of standing still debating ad infinitum. Of course, that execution speed from putting on blinders means that the team might miss an existential threat just out of their field of view.

Optionality is not free; it makes every decision at the macro and micro level more difficult and can make coordination costs far higher. Like all things, finding the proper balance of breadth of optionality vs precision of focus is a contextual balance. In practice, organizations often reserve far too much optionality, trading off theoretical optionality with concretely significantly slower execution. An organization trying to do everything well will do practically nothing at all.

Pattern: Have a compelling, visible leader

Having a compelling, visible leader makes the team more likely to pick the same option. Try the interactive version.
If there’s one person who looms especially large in the organization, because of their formal authority, connectedness in the network, credibility, or just because they’re a compelling communicator, then they will have outsize influence on the decision of the organization. The thing they think is right is far more likely to be the thing that everyone picks. This can be a bad or good thing depending on how accurate that person is, and recall that in complex problem domains no one person has all of the necessary context. The leader need not be the formal lead. Sometimes there are behind-the-scenes “influencers” who are not in visible roles but are extremely well connected and well respected across the organization. These people will often have far more influence on decisions than might be superficially obvious from studying the org chart. In this model, visible leads manifests as some individuals having much stronger connections to the rest of the team, and being more likely to broadcast to multiple people at once.

Pattern: Develop shared mental models

Shared ways of looking at the problem gives less disagreement and more alignment. Try the interactive version.
Another approach is to ensure that the collaborators across the team share mental models. Typically this takes the form of everyone being bought into some overarching strategy. It can also take the form of the group investing time to look at the problem through similar lenses, using shared terminology, working from a widely-shared base of facts, agreeing on the same time horizon to judge value, etc. This will lead to more collaborators naturally judging options similarly and be more likely to pick the same one. In this model, it manifests as smaller error bars.

Pattern: Create a convincing argument

A convincing argument makes one option stand out in sharper focus. Try the interactive version.
Sometimes a convincing communicator can construct an argument that a number of people all find plausible and convincing, leading the group to be more likely to pick that option. In this model, it’s represented as one of the options having large value and significantly smaller error bars, bringing it into sharper focus.

When this is working well, the argument is a truly rigorous one that implicitly or explicitly compares the primary option to the other plausible options. However, there are often a huge number of plausible options and it’s not possible to compare against all of them, so in practice arguments typically make a convincing case for one option and just ignore the rest.

Note that a convincing argument is not necessarily a rigorous one. Various rhetorical tricks make an argument feel more convincing than it deserves. For example, you can set up your constraints section of the proposal so that the solution section of the proposal perfectly matches it, like fingers in a glove. The elegance and fit of the solution will make it feel more convincing, even if those constraints were specifically picked to complement the proposed solution. Another trick is to figure out a viral “hook” that will stick in people’s heads and that they’ll want to share (like comparing an organization to a slime mold, or some other evocative and slightly subversive metaphor). Another trick is to use numbers to give a patina of precision to the argument. Numbers are like a comfort blanket, protecting the reader from the inherent uncertainty of complex problem domains. People will tend to find numbers convincing even if they are based on no more than napkin-sketch models, and they will cling to them as an anchor point even if they are wildly inaccurate or have a significant amount of faux precision. Be on the lookout for where you or others in the organization may have intentionally or unintentionally relied on these tricks to make one option stand out.

Pattern: Respond to an existential threat or opportunity

When there’s an existential threat, it’s easy for everyone to agree it’s most important. Try the interactive version.
Sometimes something happens in the surrounding context that makes it extremely clear that you have to do something. Maybe your competitor launches a game-changing feature that changes consumer expectations, compelling you to respond quickly. Maybe some new macro trend creates an existential threat for your organization. Decisions that previously would have taken months might now take mere days. In this model it manifests as one option being significantly higher value than all of the others.

Sometimes you find yourself in a situation where everyone fears that an existential threat will pop up — say, there are rumors that a competitor will launch a specific feature in the near future. In the lead up it can be very hard to convince people to coordinate around the idea, because the “value” of doing a project to counter a theoretical situation is extremely debatable. Instead of investing exorbitant amounts of effort to fight that entropy, create a “break glass” plan — a two or three page document describing what actions you would take if the competitor ships that feature. Those plans are way easier to create than to execute on (just ask each team what they would do if suddenly shipping a competitive feature were unambiguously the most important). Then, if the competitor ships the feature, you can jump into action with very little coordination cost because it’s now unambiguously an existential threat that everyone will take seriously. But if the competitor doesn’t ship the feature, you won’t have wasted an inordinate amount of effort fighting the coordination headwind.

Another dark pattern to watch out for with existential threats is an us-vs-them dynamic. One way to manufacture the perception of an existential threat is to get people to believe that some other group is an enemy who is seeking to destroy you or is evil and must be stopped. This can whip the team into a frenzy and make them easier to coordinate, but it’s fundamentally a toxic dynamic. This pattern happens both intentionally and unintentionally, so keep an eye out.

Pattern: Create compelling north stars

A compelling north star makes it more likely for people to pick the same project. Try the interactive version.
A north star is a long-range strategy that gives people in the organization a direction to sight off. It helps teams break ties between different options and pick the ones that add value in the short term but also bring the team closer to the north star. This is related to “reduce the number of options”, because it puts special focus on the options that are in the direction of the north star. It’s also related to “shared mental model” because everyone looks at the options in a similar way. In this model it is represented by collaborators’ baseline value estimates being tweaked based on if they are in the direction of the north star or not.

The north star must be high quality: everyone across the team should agree that it is a desirable outcome, and also that it is plausible (it doesn’t require miracles). It must also be widely communicated and understood across the team. Like any structure, it will help increase efficiency, but that structure can become a future liability if the conditions change. If things change and your north star is now pointing you in the wrong direction, the organization will continue lumbering towards the old north star like a horde of zombies. The north star will have been baked into a number of load-bearing assumptions, large and small, throughout the organization. If the north star was in place for a long time, everyone will just take for granted that the well-established north star is a fundamental truth and won’t even be able to conceive there’s another way. Getting the group to sight off of a new north star will take an enormous influx of energy.

Anti-pattern: an unconvincing north star
Note that if you have a north star that people don’t find believable, the situation will play out differently. In this model, we’ll represent it as people who secretly find the north star implausible and aren’t affected by it.


An unconvincing north star will have worse alignment. Try the interactive version.
This anti-pattern can happen when there is a lead steamrolling towards a particular north star. By making a loud, consistent push for it from a position of authority, it emerges as the only plausible schelling point. This situation can work extremely well, especially in top-down organizations or for short periods of time. But it is also prone to catastrophic failures. The lead is creating an intersubjective reality where everybody thinks that everybody thinks that the direction the lead is pushing for is the best one. But each individual might suspect that that direction is not a good one — maybe they possess information that shows it requires a number of miracles to succeed, or they think it is pushing for a small amount of value. If people gossip behind the scenes they might realize that many people are skeptical. The organization is now in a supercritical state. Superficially everything is stable and the org is pointed towards the supposed north star, but the foundations have rotted. All it takes is one inciting incident — one unguarded comment from the lead, one awkward question at an all hands, one high-profile team departure — for everyone to realize that everyone disagrees with the north star. Like the emperor who was wearing no clothes, a single inciting incident can set off a cascade where the fragile equilibrium is overturned.

Pattern: Don’t even pretend to pick the best one

Picking a randomly agreed upon project can help create more alignment. Try the interactive version.
The schelling point lens implies another counter-intuitive pattern that can work in some cases. Sometimes you find yourself in a situation where there are multiple plausible options that all seem to have roughly the same value. This can happen when you know that, over sufficiently long time horizons, you want to do all of the projects and it’s just a matter of which one to do first. You might fight forever about which one is the best, because no one of them obviously is. In those cases, don’t even pretend to select the best one. Get everyone to understand the fundamental schelling point dynamic, and that the most important thing is the group agree on the same one. Get everyone to agree on some arbitrary selection criteria, e.g. “a random die roll to select one of the ones that everyone agrees is unlikely to be particularly hard”, and agree to commit to the one that gets selected. Often once you pick one and execute on it as a team, you’ll get momentum and develop muscle memory for how to tackle this kind of problem, making tackling the next ones easier and easier. You can think of this approach like having an arbitrary but widely agreed upon north star.

A couple more tricks
That’s a tour of many of the patterns to reduce the coordination cost in an organization. We’ve only just scratched the surface of the kinds of questions we can ask by playing with the model we’ve defined — for example, how bad is it if two different well-connected leads have very different understandings of which projects create value? Check out the interactive model and experiment!

There are also a few other meta patterns that can apply in some cases that aren’t directly implied by the model we’ve created.

Splitting into multiple balance points. If you’re trying to balance a wide variety of interests, there might not be a single balance point: there might be strong forces pulling in both directions leaving no one happy. In those cases, sometimes you can split it to create multiple balance points that can balance separately. For example, perhaps your product is well known for being simple to use right out of the box, but advanced users are frustrated that they can’t configure it for their particular needs. In that case, you can could create two flavors of your product: one the simple front door that provides safe defaults, and the other the advanced, underlying platform with full power. Bonus points if you’re able to make it so the top layer is literally just “sugar” for the underlying layer, helping create a well-layered platform. The two layers might have distinct but related brands, allowing you to market to two different audiences. Another more common example of splitting balance points is to simply split a larger team into smaller teams with slightly different objectives.

Lumping multiple options together. Say that you have a product that successfully serves the needs of the biggest users — the head. When considering how to expand your product, every other use case looks small in relation to what you’re already doing, making it hard to agree on which use cases to tackle next. Sometimes it’s easy to erroneously conclude that there’s simply no more value to unlock down the tail. In these cases it can feel like if you just keep searching through the haystack you’ll find the needle use case that everyone will agree is worth it. But in many cases — like if you started in the head and are going to the tail, or if your problem space is a platform — there are no single killer use cases, meaning you’ll be searching that haystack forever! Instead of looking for individual, vertical use cases, look for aggregated, horizontal use cases, effectively lumping a number of use cases into a bundle. That will help you see the cohesive value more clearly.

The awesome power of schelling points
Schelling points can be about specific, individual projects. Or they can be about large, nebulous things like an organization’s culture. For example, once offices reopen, will a given org tend to fall into an equilibrium where most people come to the office (and other people on the edge decide to also come in to avoid being left out) or most people stay home? As another example, does an organization prefer email or Slack (where even low-priority messages need to be sent via Slack in a hope of rising above the cacophony)? Another example is a culture’s approach to giving and receiving feedback. These larger instances of schelling points are harder to grasp (or even detect in the first place) but can be extremely powerful undercurrents.

Schelling points can also be used to understand other human interactions. When we speak with others, we are trying to induce a set of thoughts into their minds. Unfortunately, we can’t just directly beam those thoughts into place; we have to communicate indirectly, by picking and choosing the right set of words — tiny packets of shared meaning — that will induce the desired understanding. It’s like shooting little peas with a peashooter into someone’s brain, hoping to knock over some thought dominoes to set off a cascade of understanding. Critically, we pick those peas based on a pre-existing schelling point of mutual understanding about what those words mean. Even people with a deeply shared context will interpret words at least a bit differently — if we didn’t, there’d be no room for language to evolve — but words have to have some kind of sufficient shared overlap of meaning in order to be useful tools to induce the desired thoughts.

Which schelling points are plausible in a given situation is highly path dependent and influenced by a the organizations’ deeply etched history. Creating a schelling point out of chaos is expensive. Even more expensive is creating a new one that fights against an already deeply ingrained one. Realize that you are not as in control as you think. The emergent schelling points are in control, and your control was always largely an illusion. Don’t spend time pining for what could have been, focus on what can be based on the constraints of where you are.

Remember that the schelling point only has to be good enough to be obviously better than other alternatives. Not much better, just obviously better. That means that schelling points can often start off as seemingly arbitrary blips.

Schelling points that emerge tend to strengthen; every incremental new collaborator looking for a direction to go will see one that everyone else is going towards and, unless they have some strong reason not to, will decide they might as well go that direction, too. This is a preferential attachment phenomenon that intrinsically creates compounding momentum. That means that schelling points that started out as arbitrary blips can bloom into broad bright beacons that coordinate the actions of large collections of people. Coordination is power, and wide-scale coordination is wide-scale power to make big things happen in the world.

All of these invisible, overlapping schelling points of varying scales combine into a force of awesome power that influences everything around us. This power is fundamentally amoral; it can cause amazing or terrible things to happen, and those influencing the forces can be acting morally or immorally depending on their intentions and the real-world impact they cause.

Let go of the idea of heroically searching the haystack for that one impossible-to-find “right” answer. Find one that’s good enough to overcome entropy and create net value, and just get going. Then, as you pick up compounding momentum, you can steer it in the direction it needs to go, to maximize value from the broadest possible perspective. Ideas are a dime a dozen, and you can debate which one is “better” until the cows come home. Compounding momentum is what changes the world, for better or worse.

In the space of possibility, the chaos of entropy is the background radiation in the fabric of spacetime. In that cacophony, every so often a little tear opens up: an ephemeral schelling point. These are sparks of possibility blipping into existence, struggling to remain viable, growing, and fizzling — innumerable times every moment. Putting a dent in the world is about sensing these sparks, giving them material to grow, fanning the flames, harnessing their energy, and arcing them towards creating value in the world. If you can do this you’ll be an alchemist, tapping into a hidden, magic undercurrent, surfing it to cause big things to happen in way that looks almost effortless.

Editor’s note: While working on this post I realized that having diagrams would help make the concepts easier to grok. After I built a few diagrams I realized they’d be more clarifying if it could show a number of different runs of each scenario in a GIF. If you’re going to generate a GIF, you might as well create some tooling to make it interactive, too. That was the start of me (predictably) falling into a hobby project rabbit hole. The project blossomed into https://github.com/jkomoros/CASsim,a hacky tool to create interactive agent based models, share them, and generate GIFs of them. It’s still very early and rough around the edges, with a lot of features I want to add. Give it a spin and let me know what you think by filing an issue or reaching out at alex@komoroske.com.

