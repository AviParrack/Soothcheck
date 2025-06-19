# 9/16/24

*I’ll be on vacation next week so there will be no Bits and Bobs. They’ll resume on Monday 9/30.*

- Software is an extension of human agency.

  - It has no agency of its own.

  - It is put in motion by the creator of the code (who created the potential energy) and the person who decided to run it this time (who converted the potential energy to kinetic energy).

    - Sometimes the creator and the runner are the same person.

    - Sometimes the code was written once, long ago, and is run trillions of time today by millions of people.

    - When code is executed, it is a vote that it is useful and worth keeping around and maintained.

  - This notion of agency affects plain old code, which does exactly what you said.

  - Code that can think for itself–like LLMs–is not a straightforward extension of human agency.

    - It can be, if it’s a minor component of a larger system, the magical duct tape.

    - But if the LLM is in the driver’s seat, if it’s treated like an oracle, then it has absorbed some of the human’s agency.

- Agency is a muscle.

  - You have to exercise it!

- A famous Murray Gell-Mann quote: “Imagine how hard physics would be if atoms could think.”

  - Programming with plain old code is powerful and pure, but also, in some way, easy.

    - The code does precisely what you told it to (though not necessarily what you meant).

  - It’s easy for programmers to look at people in other systems, less in control of the output, and look down on them.

  - But now LLMs can “think”.

  - Suddenly, corralling them becomes harder than writing code!

- Simon Willison [<u>asked on Twitter for people’s favorite LLM prompting tips</u>](https://x.com/simonw/status/1832944559162269990).

  - The answers were all over the map.

  - That implies that we’re still in the early days of collectively figuring out how best to use them.

  - In a mature phase of a technology, the most effective playbooks are widely known and no one bothers doing anything different.

- Longer feedback loops are inherently much harder to steer.

  - The way we learn is by taking an action with a predicted effect, seeing how the outcome matched the prediction, and then adjusting.

  - The longer it takes from action to effect, the longer the feedback loop cycle, and the slower the learning cycle.

  - The more action/effect cycles you get, the more you absorb the knowhow of how to steer this particular thing.

  - But the longer the feedback loop, also the more diffuse and indirect the effects of the steering, and the harder to extract the learning.

  - If there’s a short feedback loop with a very direct connection, then the outcome arises straightforwardly out of almost entirely your action.

  - But if the loop is longer, there’s more environmental factors and outside factors that affect the system from your action to the outcome.

  - Those muddle the analysis; was the effect because I steered it wrong, or because of that random gust of wind halfway through?

  - In longer feedback loops you’ll tend to oversteer and then have to steer back to get it back on track.

    - In the limit this can cause oscillations around the intended level.

  - It’s harder to extract the signal from the loop.

  - Long feedback loops are orders of magnitude harder to learn how to steer.

- The longer the feedback loop, the more annoying to use--even if the loop gives good results.

  - You get around the loop, excruciatingly, and go "no, no that's not it, here's new directions".

  - Even if the system does a good job!

  - Because maybe you steered it wrong, and you don't know until much later, and then have to go "ugh, is it worth it to wait for yet another loop through?"

- As you get more senior, the feedback loops go from one day to two years.

  - The steering wheel is connected with tons of leverage, but tons of lag.

  - Very very hard to learn in that environment.

  - So by the time you notice what it's done and realize it's wrong, you're already wildly off-track.

  - You’ll ask yourself, “who’s causing all of this thrash and wasted effort?” and the answer will be “me, months ago.”

- A few random off-the-cuff reactions to OpenAI's strawberry model in no particular order:

  - The performance of it is something that could be sometimes cobbled together with a whole lot of prompt-fu and judgment.

    - e.g. [<u>https://x.com/tommy_winarta/status/1834550186099576958</u>](https://x.com/tommy_winarta/status/1834550186099576958) and [<u>https://x.com/daveshapi/status/1834599760931569677</u>](https://x.com/daveshapi/status/1834599760931569677)

    - But now it's fully automated and doesn't require user savviness, and also it's potentially self-improving.

    - It's kind of "just make it do chain of thought, but hide the intermediate parts from the user to not distract them, and also make the problem solving parts self-improving with more training"

  - One of the reasons AlphaGo was self-improving (more compute = better skills, without limit) was because the rules of Go are well grounded.

    - It was very easy to keep it aligned with how Go actually works so as it improves it never loses contact with reality.

      - A computer could run a normal program to ensure the “laws of physics” were consistent.

  - Strawberry is *expensive*!

    - It reminds me of the "supersonic consumer air travel is physically possible but not economically viable."

    - There will be a point where for a given use case it's just too expensive.

    - So you only break out the big guns rarely and when you really need it.

  - The thing I think is most interesting is how hard it is to steer based on a very long feedback loop.

    - Set it and then come back to it later... and if you realize "oh crap I forgot to include some relevant context" you have to do it again.

    - Longer feedback loops are inherently much harder to steer, and harder to learn how to steer because you get less rounds of experience with it.

    - The dumber models require more of a human on the steering wheel in iterations to guide it... a downside is the human has to be there and has to know how to do the lightweight steering (prompt-fu) but the upside is that the human can continuously correct with small corrections instead of it running off for minutes to answer a question that turns out to be ill-posed.

- It's amazing how much competition there is for LLMs.

  - Everyone standardized on OpenAI's API, which allows fast switching between providers.

  - Whichever knowledgebase users store their data in for the LLM to use is very sticky.

  - The LLM matters less than the system that stores your data.

  - It just so happens that ChatGPT stores some state for users (e.g. memories) and is getting some stickiness… but it need not be the thing from the model providers.

  - It’s totally possible that someone could use off-the-shelf models from the top providers and create the sticky, value-generating service.

  - The LLM providers in that world would be kind of like internet providers: behind the scenes with significant competition.

- An LLM writing software for you today has to start from scratch each time.

  - If you could store the code that it had already written for it to use as building blocks, it could go much further.

  - You’d need some way of storing state in the system.

  - Even better: if the work the LLM did to write one user’s software could automatically make it better at writing *other* users’ software, too.

- If you try to fit LLMs into your existing workflow, it will fail.

  - But people who are willing to change their workflow can realize its power.

  - Programmers can change their own toys like play doh, so they are better at changing their workflow to adopt technologies earlier than others.

- Automations have to be reliable.

  - When an automation is working correctly, you don’t have to think about it.

    - Out of sight, out of mind.

  - The more that it works, the more you take it for granted, the more it fades from your awareness.

  - This frees up more of your mental effort to focus on novel, useful things.

  - This is good!

  - But it also means you’re getting set up for a rude awakening.

  - If the automation fails, the problem crashes back into your awareness and demands all of your attention.

  - It’s an emergency, and you don’t have the context of how the automation works loaded up in your brain.

  - A stressful situation!

- Without the internet it would be impossible to have LLMs.

  - The "assembly condition" of LLMs is language --\> books -\> internet -\> LLMs.

  - LLMs could not exist without any of those steps in that chain.

- Desktop OSes were data centric. Mobile OSes are app centric.

  - Data centric means that there’s a shared filesystem that applications write to and read from.

    - There’s a default application for most filetypes, but you can open many files with many different applications.

    - New applications can show up to innovate on existing data.

  - App centric means that the data doesn’t exist outside the context of a given app.

    - If you uninstall the app your data evaporates.

    - Data does not make sense as a concept outside an app, with limited exceptions like a photo roll.

    - A user can only access their data in an app on terms the app defines.

    - Don’t like what the app does? Too bad!

  - A data-centric OS allows data to be reused in novel ways.

  - Let’s return to a data-centric world.

  - Where a user’s data is the center of their universe.

- The job of a platform is to solve the shared, boring, hard stuff so that developers don't have to think about it any more.

  - This lifts up what everyone can do, allowing developers’ effort to go towards the interesting and novel parts of their problem.

- Lofi systems encourage tinkering.

  - If you're a tinkerer putting the first 3D object you made into a world with AAA 3D assets your thing will look embarrassing.

  - A lofi system has a lower bar to get started, and once people get started there's often a gradient to continually improve.

  - A key dimension of a generative system: how hard is it to get a "good enough" creation?

- Front-end development nowadays is largely about monoliths.

  - Single Redux stores.

    - Even if some of the UI is componentized, the state management is not.

    - Any component that is connected to the store is not possible to reuse in a different context.

  - Knowledgeable experts understand monoliths better.

    - They can invest the time to understand it, and when they do the whole system is knowable.

  - Tinkerers understand a modular system better.

    - They can incrementally understand just the small pockets they need to, and they would have never understood the whole system anyway.

  - For a new kind of tinkerable software there should be a return to modularity in frontends.

- If an LLM could do it, then don't form a company on that thesis!

  - Not only will other competitors sprout up to do it better, but users can just do it themselves!

  - People are calibrated based on what would have been hard to do in the past.

  - But now *everyone* has LLMs.

  - It's not a cheat code that only you figured out.

  - The "be the first, and then get a compounding advantage" only works if it's hard to build what you built.

  - LLMs make it easy for you... but also everyone else!

- [<u>I’ve talked in the past</u>](https://docs.google.com/presentation/d/1KQrs3s1LTBD7RKhNokx94VfR5UQnnSsUewiLYYXKRQA/edit#slide=id.g2e35f4658a5_0_661) about a pattern I might call the Negentropic Wave.

  - You create an auto-catalyzing wave of energy to surf to your destination.

  - You do this by finding the items that in the background brownian motion just so happen to be headed in the direction you want to go, and subset them into a new collection.

  - The momentum of the collective will then be strongly in the direction you want to go… and you never had to break a sweat, you just applied a curatorial selection to what already exists.

  - Why call it a negentropic wave?

    - The mechanism of action is similar to Maxwell's demon thought experiment… except at macro scales, it works!

    - Maxwell’s demon thought experiment is about entropy.

    - Schroedinger noted that some systems, like living systems, seem to become more ordered over time.

    - He called these systems “negative entropy”, or Negentropy.

  - Imagine applying this technique within an organization, to find a set of volunteers on a project you think will change the game.

  - One might ask: “how do you keep them motivated?”

  - But the trick is that they were already intrinsically motivated to do this work before you found them.

  - So you don’t have to convince them to do something they don’t want to do.

  - All you have to do is fan the flames of energy they already had.

  - This is easy! Especially if they feel like that energy is part of a collective that is gaining momentum.

  - If you were trying to push via extrinsic energy, then if you got distracted the thing would run out of steam.

  - But a negentropic wave is one that is auto-catalyzing even if you get distracted.

  - Even when you aren’t looking, it will continue going.

- Don't force people to work on things they don't get energy from.

  - It's a form of torture that slowly saps their life force.

- A secret is a thing others think is impossible but you know is actually possible.

  - If you can sequence multiple secrets, no one else will even try to follow you.

  - Every time someone hits a "that's impossible" in a sequence they stop.

  - Sometimes they actually are possible, just requires doing something out of the ordinary, lateral thinking.

  - If your idea requires three of those miracle points that look impossible but actually aren’t, then the likelihood someone else sees it is very low.

- Innovation tokens in serial is dangerous but in parallel is good.

  - Think of an innovation token as a novel problem.

    - A high chance that it doesn’t work, but a low chance that it does work.

    - And if it does work, it’s game-changing.

  - In serial: you need all of them to work, and the likelihood that *all* of them work diminishes super-linearly.

  - In parallel: if any one of them works, you get something game changing.

    - Each one gives you *more* chances for a game-changing result.

  - Innovation tokens are acorns.

- In a stampede, the person who stops and says "wait, why is everyone running?" gets run over.

- One reason the internet grew as quickly as it did was because the capex was at the edge.

  - On the edge, the marginal investment (e.g. buying a modem) led directly to marginal value for the investor.

  - With investment in the center, there's at least one extra ply.

    - You invest, then someone else does something *then* value is created.

    - More indirect, less sure, a slower loop.

    - Systems with longer feedback loops grow and adapt much more slowly.

- Magic that makes your system harder for users to reason about is a curse.

- A community with no one talking is dead.

  - A community with only one person talking doesn't realize it's dead yet.

- Look for ideas that robustly resonate.

  - Robustly resonating: nearly everyone who hears it is intrigued.

  - The more diversity of people you find who resonate with it, the better: the more likely that the next person you talk to also resonates.

  - An idea that many people from many different perspectives have an “aha” moment for is more likely to be interesting: surprising and potentially valuable.

  - If you find an idea that resonates with just about everyone, it’s a great idea, with the potential to go viral and change a lot of beliefs quickly.

- Risk only really shows up in larger time slices.

  - If you can slice things down to fit in the adjacent possible, there isn't that much risk.

  - That's why the [<u>iterative adjacent possible</u>](https://komoroske.com/iterative-adjacent-possible) is so powerful: no risk, but same upside.

  - Take each step and then just keep taking them while it continues to work.

  - You don’t even have to know if steps many steps down the line will work: it doesn’t matter!

  - When considering a next step, ask yourself:

    - 1\) Will it almost certainly work and pay for itself?

      - Either because it creates value, or the cost is low.

      - For example, a thing that you enjoy doing and gives you energy has a negative opportunity cost.

    - 2\) Is there a chance that down that path is a great (as opposed to merely good) outcome?

      - Don’t get too hung up on precision for the “great” outcome–if it’s compounding, or multiple orders of magnitude beyond what you can do today, that’s fine.

      - Don’t spend time getting faux precision.

  - If those two things are “yeah” then stop navelgazing and just do it!

  - Then, repeat!

- It can be scary to share your thing with the world.

  - That’s the moment of the truth: when the idea is ground truthed, and shown to be viable or non-viable.

  - The longer you’ve been working on the idea in a cave, the more likely the idea is non-viable.

    - Perhaps you’ve lost touch with the ground truth as you iterated in the darkness.

  - It’s much better to have your thing interact with the ground truth early and continuously.

  - When you’re in the cave, the value of each additional unit of discussion diminishes logarithmically.

    - It gets increasingly disconnected from the ground truth.

    - The quality of the insights based on the limited data already brought into the discussion from outside the cave improves slower and slower.

    - Another unit of time spent talking about information you’ve already collected is much less valuable than a unit of time of sourcing new disconfirming evidence by getting more information from outside the cave, or even better, sharing the idea in the outside world and seeing if it’s viable.

  - It can be very scary to share your idea with the outside world, since it might fall on its face and die.

  - But it doesn’t have to be that big of a deal.

  - There are often ways to float trial balloons to get feedback before everyone sees it.

  - Conceptually, you’re trying to minimize the chance that someone who uses it has such a bad time they never use it again.

  - One way to minimize that is to have the quality be perfect… hard to do when working in a cave.

  - Another way is to control your audience so only a small number of resilient people use it.

  - The default way to do that is to pick your audience and tell them you think they should try it.

  - The other way to do it is to have a gauntlet that leads to motivated users self-selecting.

    - E.g. have minimal documentation, make the tool look a little sketchy.

    - The users who are willing to put up with the inconvenience in the first steps will be more likely to be resilient to a broken experience in the actual product.

    - And then if they like it, you can ramp up your outbound marketing, and reduce the broken glass to expand to a larger audience.

    - You get the benefit of the upside if it’s good, and self-capping downside if it’s not.

    - This lets you sample how possible users in the ecosystem feel, sensing ahead of time if there is a wave to catch before putting yourself out there.

  - Another example of self-capping downside is when there’s some high-stakes decider who needs to weigh in.

    - For example, an investor you want to invest, or your superior in an organization who will decide if the project goes forward.

    - If you go in and formally have them weigh in and they don’t like it, it could be game over.

    - But imagine that that decider is someone who likes you well enough that they’re willing to meet with you 1:1 well before any formal meeting.

    - In the meeting, you ask for mentorship, e.g. “What kinds of things would let you know that an idea was ready to go.”

    - This is effectively a trial balloon. If it goes well, you keep pushing the advantage, and maybe even lead to a formal-ish decision in that meeting.

    - If it doesn’t go well, that’s OK, you were just asking for mentorship anyway, and you can pull back on that line of discussion as soon as you hit bumps in the road.

      - Just make sure you have enough enjoyable / useful agenda items to run out the clock without them feeling like it was a waste of time.

- Find the thing that is in your adjacent possible but not in others’.

  - Lean into your differentiation

  - Others will say "is he magic? how did he do that?" but the magic is just the clarity to see what is special in your adjacent possible, the taste to select it.

- The [<u>hourglass pattern</u>](#e7ty1ew9ifa6) only works if you have multiple options at the top and bottom layers.

  - If you only have one option at the top part of the hourglass, you aren't sure that the neck of the hourglass is actually general purpose enough to work for different layers on top.

  - It's only by exercising it with multiple options that you can verify that it's not overfit to any one.

- You have to implicitly answer "why now" when prioritizing a task among all the other things you could do.

  - There are two very good answers to this:

    - 1\) "because it just happened and it's fresh in my brain"

    - 2\) "because there is an external deadline coming up and I'm obliged to."

  - People prone to procrastinating work best under the latter.

  - But if the task is small, will definitely need to get done at some point, and it just happened, just do it now.

    - If you don’t it will linger and fester and be thrown on top of an ever-more amorphous, ugly, intimidating pile that will take heroics to fix.

- Prototypes should be at the highest pace layer.

  - The most important thing for a prototype is adaptability; finding the fun.

  - Ideally prototyping in a team should all happen on the same pace layer.

  - If there’s one prototype being developed on multiple pace layers, you have two bad options:

    - 1\) the prototype goes as slowly as the lowest pace layer, making it significantly less likely to find the viable product sketch.

    - 2\) the engineer responsible for the lowest pace layer tries to keep in sync with the faster moving pace layer through sheer force of will.

      - Like spiderman holding the two halves of the ferry together.

      - Impossibly stressful, if you take a break for a second you rip apart.

- Taste is having a perspective that stands out from the average and people find compelling.

- Taste has always been the most important, but it wasn’t always obvious.

  - When production is expensive and requires skill, the importance of taste is obscured.

    - Because to create something you have to have taste *and* the ability to produce.

  - But in an era of fast content production, taste remains as important.

  - If you have good taste, LLMs can now allow you to get that stuff generated 10 times faster.

  - With tools like generative AI, the difference between curation and creation will get even smaller.

- A recipe for game-changing, wonderful things: scrappiness and taste.

  - Scrappiness allows you to try ideas quickly and find good ones.

  - Taste means that the ideas you find, even if a rough sketch, are likely to be compelling.

- Imagine a situation that is worth either nothing or has extraordinary results.

  - This binary outcome makes the decisions way easier to make.

  - You don’t need to calculate expected value or anything.

  - Just consider the null outcome and the great outcome, and see how bad it would be in either case.

  - For example, maybe it’s the null outcome… but you enjoyed the work and the people you worked with and it made your resume more impressive. In that case, who cares?

  - No downside and non-trivial chance of massive upside is a no brainer.

- Narcissists can’t see other people’s pain.

  - This means that they are more likely to steamroll forward on ideas that are good for them.

  - The steamroller is running over people who are screaming in pain… but the narcissist can’t hear it.

  - Asked to justify what they’re doing, they say “here’s why it’s good for me,” completely oblivious to the downside for others.

  - Because the downside affects other people, and is literally invisible to them.

- Small talk is boring. Big talk is illuminating.

  - People will rise to the level of interestingness of the interest their conversation partner takes in them.

  - If you're interested, the things around you will be interesting.

  - Everyone has an interesting thing to say, if only you're interested enough to hear it.

  - Big talk happens when both sides of the conversation are interested in one another.

  - If only one side is ready for big talk, then it will collapse to small talk (outside of heroic illumination by the one side).

  - Be open to big talk happening in any conversation you’re in and the world will reward you with interesting things to hear.

- Being meaningful and being engaging are distinct.

  - Engaging means fun in the moment; enjoyable for its own sake.

  - Meaningful means that afterwards you think “I’m proud I did that”

  - Not engaging and not meaningful: waiting in line at the DMV.

  - Engaging but not meaningful: an addictive mobile game.

  - Meaningful but not engaging: volunteering to do menial tasks for a cause you believe in.

  - Engaging and meaningful: big talk with an important person in your life.

  - It’s possible to find meaning in just about anything if you’re open to it.

  - If you find something that’s both engaging and meaningful, pour your heart into it.

- The set of people who can verify "yes, that trophy is valuable" is way larger (and less motivated) than the set of people who will sift through the firehose to find those trophies.

  - That's part of why you see a power law distribution of engagement in any ecosystem.

  - The "spoils" are "how many people think more highly of you given you found the trophy" and the difficulty is "how much effort does it take to find a trophy".

- If a viral and antiviral thing compete the viral thing will dominate even if no one wants it to.

  - Virality means there will be more copies of it in the next time step.

  - Engagement is about virality; meaning is about whether or not we’d be proud of the outcome.

  - “People should do this meaningful thing, not this engaging thing!” is a losing proposition.

  - Instead, find the ways to make the meaningful thing engaging too.

- The point of surfing is the surfing, not where you get to.

  - If you're making a product that feels like surfing, focus on making the surfing fun.

- A parable invites you to think, grapple with the ambiguity, engage with it.

  - It doesn't give you a pat answer, it gives you a generative set of questions for you to make your own meaning.

  - Meaning you construct yourself is way richer and more powerful than meaning you were handed.

- Last week I [<u>talked about seeds, saplings, and trees</u>](#93anq2mvxwn).

  - A seed is something that a single person, or a small group, can do entirely in "extra" time on top of their normal responsibilities.

  - A seed only takes the intrinsic motivation of a person or group of people to begin growing.

  - Skunkworks are more saplings.

    - If it takes extrinsic investment that competes with their day job, then it's a sapling.

- Being the primary owner of a thing requires constant awareness of the thing.

  - Constantly asking "is now the time to jump into action?”

    - Constantly looking out for evidence that some action needs to be taken.

    - A constant busy loop, refreshing the awareness in your head and asking, "now?".

  - Whereas being in reactive mode is orders of magnitude easier.

    - If there is something to do, the world will let you know.

    - Until that point, out of sight, out of mind.

    - No wasted mental effort.

  - There are only a limited number of things you can be a DRI for, but you can be a reactive participant in many orders of magnitude more things.

- When you know your thing might be received negatively, inoculate the reader right at the front.

  - This will make them more open to receiving it.

  - For example, say “What I’m going to say will sound cynical, but I think it’s actually optimistic. By understanding how the system *actually* works, with clear eyes and without judgment, we can see how to steer it to great outcomes.”

  - Now later, when you say something cynical, instead of them saying “this guy sounds cynical,” and tuning out, they say “well he did say he’d *sound* cynical, so I’ll keep listening”

  - You’ve effectively capped the downside of people disengaging from your argument, but preserving the upside.

  - If the person agrees with your warning, they’ll think “well he did tell me, at least he’s self-aware about it.”

  - And if the warning was unnecessary because they agreed with what you said, they’ll forget about the warning.

  - So the warning has no downside and significant upside.

  - That means the best practice is to imagine what the negative reaction is that listeners are most likely to have and explicitly inoculate that.

- When there’s a scarce resource people want, they’ll fight over it, even if you don’t see them fighting.

  - Imagine you are a very powerful person.

  - Everyone would love to get your attention and be in your good graces.

  - Most of that jockeying for position will be completely outside your view.

  - If you don’t think about it much, you won’t even realize it’s happening.

    - Out of sight, out of mind.

  - In that position, even tiny tweaks in how you spend your time can have huge repercussions.

    - For example, maybe there is a person you used to work closely with.

    - But then they said something in a meeting that you didn’t agree with.

    - The project ends, and you don’t see that person very much anymore.

    - But the gossip spreads: “He iced out Sarah because she told him a thing he didn’t like!”

    - To your perspective no such thing happened: you’re simply busy and haven’t had a need to talk to Sarah recently.

  - These are the kinds of weird eddy currents of power dynamics that are inescapable in large organizations, especially for very powerful people.

  - Everyone will be scrambling to give you precisely what you want.

    - In the limit, that includes telling you white lies to your face.

    - If you’re powerful, you’ll find the confirming evidence you’re looking for… and if it doesn’t exist, it will be created for you without you having to ask for it to be.

    - Everyone will be scrambling to curry good favor.

  - This dynamic scales with how scarce the resource is.

    - For an important person, the resource is your time, and how many other people are competing for it.

- Not all blindspots are passive things.

  - Some are active, lumbering beasts, laboring outside your notice to block the view of things that could harm your ego.

  - If your ego finds it threatening, your body will throw up blindspots so you literally cannot see the threatening thing.

  - Your body will do this even when you don’t ask it to.

  - Which means you won’t realize it’s happening.

  - If you’re very powerful in an organization, the organization will scramble (without you even noticing) to give you exactly what you want.

    - The blindspot is an emergent thing, being constructed and strengthened by the organization for you.

- A big bureaucratic system is an inhuman system even though it’s made up of humans.

  - You talk to a human in the system and you think they’ll be reasonable because they’re a human.

  - But in that role they’re actually a cog in the machine.

  - The machine is perfectly content to do unreasonable things according to its internal logic.

  - If you fit with a machine, you will gradually graft more closely to it, losing your humanity.

    - LIke Bill Turner in the Pirates of the Caribbean on Davy Jones’ ship.

    - The only way to retain your humanity is to move away from the machine before it hollows you out.

    - But it’s totally fine to go to *another* machine.

    - The problem is not being part of a machine.

    - The problem is never moving from one particular role within one particular machine, since you’ll graft into it.

  - If you don’t fit with the machine, then you will not change the machine, the machine will grind you to pieces.

    - If you don’t fit with the machine you can turn off your agency and mesh with it, or you can leave.

- Two competing forces within a company: the need to compete in the market and the internal petty politics.

  - The need to compete in the market is the ground truth; if the company doesn’t create value and successfully compete against competitors it will not survive.

    - It’s a grueling slog, and it can be painful, but this competition is what creates value for the collective.

  - The internal petty politics arise, fundamentally, out of the need to not make your boss look stupid.

    - If you make your boss look stupid, you’re way more likely to be managed out.

    - So you’ll be enthusiastic about your boss (and your boss’s boss, and onwards) plan, even if you don’t think it will work.

    - Internal petty politics (kayfabe) will grow to take all available space.

  - In most companies, the kayfabe can only grow so much, because the company has to keep competing to stay alive.

  - But if a given company has no real competition, if they are a monopoly, then this petty internal politics and creeping bureaucracy has no counter; the organization becomes an ever more zombie-like organization, creating less and less value.

  - When a big monopoly doesn't innovate due to coordination costs, if they capture regulators, it makes it so the whole market also doesn't innovate.

  - Society does better when organizations need to compete to stay healthy and strong.

- Imagine you discover a deep system insight and it's so clear to you.

  - You want to share it with everyone you talk to.

  - But to everyone else you just look like a conspiracy theorist with a pinboard and yarn.

  - Even when you show it directly to them, they won't see it.

  - Every multi-ply insight looks like a conspiracy theory to someone who doesn't have the time, inclination, or openness to absorb and understand it.

  - And all systems insights are fundamentally multi-ply.

- Write down things that weren't obvious to you to start but are now.

  - These are the wedge insights that if other people understand they can shortcut the work you had to do to get to the insight and just do it.

  - Humanity is all about accumulating these shortcuts, these insights, so that we can think further and further than anyone before.

  - We do this by standing on not the shoulders of giants, but the big emergent scrap heap of useful insights accumulated via a bottom-up process called science.

  - Even things that are not formally science are still replicated when they're useful, just more informally.

    - Insights that resonate with people, that are efficient packages for communicating a deep insight, are more likely to be kept around and shared.

  - As humanity we're constantly sifting through a lot of ideas to find the set that are most clarifying.

- When there’s a boundary there’s an automatic “us vs them”.

  - "Us" is the side of the boundary I'm on, and "them" is the other side.

  - An API between teams creates a boundary.

    - In ambiguity it’s tempting as an engineer to put up an API as a kind of defensive wall to keep the ambiguity at bay, to make the ambiguity the other side’s problem.

    - “They didn’t set the requirements right” vs “*we* didn’t set the requirements right”.

      - A shared obligation vs one to put on the other and then blame them for if it doesn't work.

    - But the overall system will succeed or fail based on whether the API is the right one; whether it’s simplifying but also adaptable.

    - A good pattern to counteract this is to take the API boundary internal to a team.

    - So one person, or one team, is responsible for both sides of the API boundary.

    - This helps ensure that the API is well designed.

    - Kind of similar logic to getting a fair split of something: have one person split the item, and the other person pick. Aligning the incentives for maximum fairness.

- Humans treat “us” and “them” very differently.

  - With “us” we have much higher trust.

    - We’re willing to interpret any ambiguity in the best light.

    - We see that sometimes we do bad things, but only because the context forces us to.

  - With “them”, we have much lower trust.

    - Any ambiguity we assume is evidence of the worst possible explanation.

    - When they do bad things, it’s because of their intrinsic qualities.

  - The fundamental attribution error shows up because we think about us differently than them.

  - Watch carefully for when you–and others–use the words “us” and “them”.

    - Try to be in an expansive mindset of “us” whenever you can.

    - To constantly try to extend the boundary of “us”, bringing in more disconfirming evidence to make “us” stronger.

    - After all, all of us are shared passengers on this spaceship earth.